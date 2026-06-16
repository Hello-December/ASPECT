from ultralytics import YOLO

# 初始化YOLO模型
model = YOLO("yolo26s-pose.pt")

# 设置模型保存路径
model_save_dir = "./models/D4PED/"

# 训练配置
train_results = model.train(
    # 数据集配置
    # data=r"/home/chenxuyang/datasets/103_spacecraft/v2/dfh_4_speedplus_v4_aug_2/dongfanghong_4_coco.yaml",
    data=r"/home/chenxuyang/datasets/103_spacecraft/v2/dfh_4_dynamics_v12_split_aug_2/dongfanghong_4_coco.yaml",
    single_cls=True,
    
    # 训练参数
    epochs=100,
    device=[2, 3],
    batch=24,
    workers=64,
    
    # 图像参数
    imgsz=1024,
    pose=24,
    
    # 数据增强参数
    mosaic=0,
    close_mosaic=0,
    rect=False,
    flipud=0.2,
    fliplr=0.0,
    scale=0.2,
    hsv_v=0.6,
    augment=True,
    
    # 优化器参数
    optimizer="auto",
    warmup_epochs=3,
    warmup_bias_lr=0.015,
    
    # 其他配置
    project=model_save_dir,
    name="yolo26s_D4PED_dynamics_train_v1_finetune_0",
    amp=False,
    cache=False,
)

# 导出模型为ONNX格式
model.export(format="onnx")