# for VSCode
import sys
sys.path.append("/home/chenxuyang/PythonProjects/ASPECT/")

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePreprocessor(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=256):
        super().__init__()
        # 处理单个特征图
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)

    def forward(self, feature):
        # feature是单个张量：[B, C, H, W]
        x = F.relu(self.conv1(feature))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 统一尺寸到[B, hidden_dim, 32, 32]
        x = self.flatten(x)  # [B, hidden_dim * 32 * 32]
        x = self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))  # 支持最多100帧
        nn.init.trunc_normal_(self.pos_encoder, std=0.02)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src形状: [seq_len, batch_size, input_dim]
        seq_len, batch_size, _ = src.size()
        
        # 输入投影
        src = self.input_proj(src)  # [seq_len, batch_size, d_model]
        
        # 添加位置编码
        src = src + self.pos_encoder[:, :seq_len, :].repeat(batch_size, 1, 1).transpose(0, 1)
        src = self.dropout(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        return output

class KeypointHead(nn.Module):
    def __init__(self, d_model, num_keypoints=8, num_dimensions=2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_keypoints * num_dimensions)
        self.num_keypoints = num_keypoints
        self.num_dimensions = num_dimensions
    
    def forward(self, x):
        # x形状: [batch_size, d_model]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # 调整为[batch_size, num_keypoints, num_dimensions]
        x = x.view(-1, self.num_keypoints, self.num_dimensions)
        # 使用sigmoid确保坐标在0-1范围内
        x = torch.sigmoid(x)
        return x

class PoseTransformer(nn.Module):
    def __init__(self, feature_channels=512, hidden_dim=256, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        
        # 特征预处理
        self.preprocessor = FeaturePreprocessor(feature_channels, hidden_dim)
        
        # 计算预处理后的输入维度
        preprocessed_dim = hidden_dim * 32 * 32  # 单个特征图
        
        # Transformer编码器
        self.encoder = TransformerEncoder(preprocessed_dim, d_model, nhead, num_layers)
        
        # 注意力池化层，用于整合所有时间步的特征
        self.attention_pool = nn.MultiheadAttention(d_model, nhead)
        self.attention_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 关键点输出头
        self.keypoint_head = KeypointHead(d_model)
    
    def forward(self, features):
        """
        features: 特征张量，形状为 [batch_size, seq_len, channels, height, width] 或 [seq_len, channels, height, width]
        """
        # Check if input is a tensor
        if isinstance(features, torch.Tensor):
            # 检查输入维度
            if len(features.shape) == 4:
                # 输入形状为 [seq_len, channels, height, width]，添加batch_size维度
                features = features.unsqueeze(0)  # 变为 [1, seq_len, channels, height, width]
            
            batch_size, seq_len, channels, height, width = features.shape
            
            # 处理每一帧的特征
            processed_frames = []
            for i in range(seq_len):
                # 获取当前帧特征 [batch_size, channels, height, width]
                frame_feature = features[:, i, :, :, :]
                # 预处理单帧特征
                processed = self.preprocessor(frame_feature)
                processed_frames.append(processed)
            
            # 将帧序列转换为Transformer输入格式 [seq_len, batch_size, input_dim]
            src = torch.stack(processed_frames, dim=0)  # [seq_len, batch_size, input_dim]
        else:
            # 兼容旧的列表输入格式
            if not features:
                raise ValueError("features不能为空")
            
            # 处理每一帧的特征
            processed_frames = []
            for frame_feature in features:
                # 预处理单帧特征
                processed = self.preprocessor(frame_feature)
                processed_frames.append(processed)
            
            # 将帧序列转换为Transformer输入格式 [seq_len, batch_size, input_dim]
            src = torch.stack(processed_frames, dim=0)  # [seq_len, batch_size, input_dim]
        
        # Transformer编码
        encoded = self.encoder(src)  # [seq_len, batch_size, d_model]
        
        # 使用注意力池化整合所有时间步的特征
        # 创建查询向量（使用最后一帧作为查询）
        query = encoded[-1, :, :].unsqueeze(0)  # 使用最后一帧作为查询 [1, batch_size, d_model]
        
        # 注意力池化：查询所有时间步的特征
        attended, _ = self.attention_pool(query, encoded, encoded)
        
        # 残差连接和层归一化
        combined = self.layer_norm(query + attended)
        
        # 投影到输出维度
        context_vector = self.attention_projection(combined.squeeze(0))  # [batch_size, d_model]
        
        # 预测关键点
        keypoints = self.keypoint_head(context_vector)
        
        return keypoints