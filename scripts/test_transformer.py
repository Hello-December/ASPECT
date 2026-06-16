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
    
    def forward(self, feature):
        # feature是单个张量：[B, C, H, W]
        x = F.relu(self.conv1(feature))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 统一尺寸到[B, hidden_dim, 32, 32]
        x = self.flatten(x)  # [B, hidden_dim * 32 * 32]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))  # 支持最多100帧
        
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
    
    def forward(self, features_list):
        """
        features_list: 列表，每个元素是一帧图片的特征张量
                      例如：[feat0, feat1, ..., featT]
                      其中feat.shape = [1, 512, 88, 128]
        """
        if not features_list:
            raise ValueError("features_list不能为空")
        
        # 处理每一帧的特征
        processed_frames = []
        for frame_feature in features_list:
            # 预处理单帧特征
            processed = self.preprocessor(frame_feature)
            processed_frames.append(processed)
        
        # 将帧序列转换为Transformer输入格式 [seq_len, batch_size, input_dim]
        # 注意：这里batch_size=1，因为每张图片是独立处理的
        src = torch.stack(processed_frames, dim=0)  # [seq_len, 1, input_dim]
        
        # Transformer编码
        encoded = self.encoder(src)  # [seq_len, batch_size, d_model]
        
        # 使用注意力池化整合所有时间步的特征
        # 创建查询向量（可以使用可学习的查询或最后一帧作为查询）
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

# 测试代码
if __name__ == "__main__":
    # 创建模拟数据
    # 模拟3帧图片的特征，每帧只有一个特征图 [1, 512, 88, 128]
    frame1_feat = torch.randn(1, 512, 88, 128)
    frame2_feat = torch.randn(1, 512, 88, 128)
    frame3_feat = torch.randn(1, 512, 88, 128)
    
    features_list = [
        frame1_feat,
        frame2_feat,
        frame3_feat
    ]
    
    # 创建模型
    model = PoseTransformer()
    
    # 测试前向传播
    with torch.no_grad():
        keypoints = model(features_list)
    
    print("模型输出形状:", keypoints.shape)
    print("输出关键点:")
    print(keypoints)
    print("\n输出格式验证:")
    print("- 批次大小:", keypoints.shape[0])
    print("- 关键点数:", keypoints.shape[1])
    print("- 坐标维度:", keypoints.shape[2])
    print("- 坐标范围:", torch.min(keypoints).item(), "到", torch.max(keypoints).item())


# 训练示例代码
def train_example():
    """
    训练PoseTransformer模型的示例代码
    """
    import time
    
    # 训练参数设置
    epochs = 10
    learning_rate = 1e-4
    batch_size = 2
    seq_length = 3  # 每批处理3帧图片
    
    # 创建模型
    model = PoseTransformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("开始训练...")
    print(f"设备: {device}")
    print(f"学习率: {learning_rate}")
    print(f"总轮数: {epochs}")
    print(f"序列长度: {seq_length}")
    print("="*50)
    
    # 训练循环
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        # 创建模拟训练数据 (batch_size个样本)
        for batch in range(batch_size):
            # 生成一个样本：seq_length帧图片的特征
            features_list = []
            for frame_idx in range(seq_length):
                # 模拟YOLO提取的单个特征图 [1, 512, 88, 128]
                feat = torch.randn(1, 512, 88, 128).to(device)
                features_list.append(feat)
            
            # 生成模拟标签 (归一化坐标，与模型输出格式一致)
            # 假设每个样本有8个关键点
            target = torch.rand(1, 8, 2).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(features_list)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均损失
        avg_loss = train_loss / batch_size
        
        # 打印训练信息
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
    
    print("="*50)
    print("训练完成!")
    
    # 保存模型
    model_path = "/home/chenxuyang/PythonProjects/ASPECT/scripts/pose_transformer_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载模型示例
    print("\n加载模型示例:")
    loaded_model = PoseTransformer()
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.to(device)
    loaded_model.eval()
    
    # 验证加载的模型
    with torch.no_grad():
        # 创建测试数据
        test_features = []
        for _ in range(seq_length):
            feat = torch.randn(1, 512, 88, 128).to(device)
            test_features.append(feat)
        
        test_output = loaded_model(test_features)
    print(f"加载的模型输出形状: {test_output.shape}")
    print(f"加载的模型输出关键点: {test_output.cpu().numpy()[:, :2, :]}")  # 只显示前2个关键点


if __name__ == "__main__":
    # 原始测试代码
    # 创建模拟数据
    # 模拟3帧图片的特征，每帧只有一个特征图 [1, 512, 88, 128]
    frame1_feat = torch.randn(1, 512, 88, 128)
    frame2_feat = torch.randn(1, 512, 88, 128)
    frame3_feat = torch.randn(1, 512, 88, 128)
    
    features_list = [
        frame1_feat,
        frame2_feat,
        frame3_feat
    ]
    
    # 创建模型
    model = PoseTransformer()
    
    # 测试前向传播
    with torch.no_grad():
        keypoints = model(features_list)
    
    print("模型输出形状:", keypoints.shape)
    print("输出关键点:")
    print(keypoints)
    print("\n输出格式验证:")
    print("- 批次大小:", keypoints.shape[0])
    print("- 关键点数:", keypoints.shape[1])
    print("- 坐标维度:", keypoints.shape[2])
    print("- 坐标范围:", torch.min(keypoints).item(), "到", torch.max(keypoints).item())
    
    # 运行训练示例
    print("\n" + "="*60)
    print("训练示例代码")
    print("="*60)
    train_example()
