#!/usr/bin/env python3
"""
CUB-200 关键点检测测试脚本
在随机10张CUB-200图片上测试训练好的模型
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 配置
CUB_DATASET = "/Users/jameszhenyu/Documents/Development/CUB-200/CUB_200_2011/CUB_200_2011"
MODEL_PATH = "models/cub-200/resnet_densenet_resources/saved_models/resnet50_wing_lr5e-05_h512_do0.2_heavy_plateau_best.pth"
OUTPUT_DIR = "/Users/jameszhenyu/Desktop/cub-test"
NUM_IMAGES = 10
IMG_SIZE = 416

# 关键点名称和颜色
PART_NAMES = ['左眼', '右眼', '喙']
PART_COLORS = ['#FF0000', '#00FF00', '#0000FF']  # 红、绿、蓝


class PartLocalizer(nn.Module):
    """关键点定位模型"""
    def __init__(self, backbone='resnet50', num_parts=3, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.num_parts = num_parts

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'densenet':
            self.backbone = models.densenet121(weights=None)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.coord_head = nn.Linear(hidden_dim // 2, num_parts * 2)
        self.vis_head = nn.Linear(hidden_dim // 2, num_parts)

    def forward(self, x):
        features = self.head(self.backbone(x))
        coords = torch.sigmoid(self.coord_head(features)).view(-1, self.num_parts, 2)
        vis = torch.sigmoid(self.vis_head(features))
        return coords, vis


def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"📦 加载模型: {model_path}")
    model = PartLocalizer(backbone='resnet50', num_parts=3, hidden_dim=512, dropout=0.2)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 处理 checkpoint 格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"   📊 加载自 epoch {checkpoint.get('epoch', '?')}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✅ 模型加载成功")
    return model


def get_random_images(dataset_path, num_images=10):
    """从数据集随机选择图片"""
    images_dir = os.path.join(dataset_path, "images")
    all_images = []
    
    for class_dir in os.listdir(images_dir):
        class_path = os.path.join(images_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(class_path, img_file))
    
    print(f"📁 数据集共有 {len(all_images)} 张图片")
    selected = random.sample(all_images, min(num_images, len(all_images)))
    print(f"🎲 随机选择 {len(selected)} 张图片")
    return selected


def preprocess_image(image_path, img_size=416):
    """预处理图片"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (w, h)
    tensor = transform(image).unsqueeze(0)
    
    return tensor, image, original_size


def draw_keypoints(image, coords, vis, original_size, img_size=416):
    """在图片上绘制关键点"""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    # 计算缩放比例
    scale_x = w / img_size
    scale_y = h / img_size
    
    results = []
    for i, (name, color) in enumerate(zip(PART_NAMES, PART_COLORS)):
        x = coords[i, 0].item() * img_size * scale_x
        y = coords[i, 1].item() * img_size * scale_y
        visibility = vis[i].item()
        
        results.append({
            'name': name,
            'x': x,
            'y': y,
            'visibility': visibility
        })
        
        # 只绘制可见性 > 0.5 的关键点
        if visibility > 0.5:
            radius = max(3, min(w, h) // 50)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline='white')
            
            # 绘制标签
            try:
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", size=max(12, min(w, h) // 30))
            except:
                font = ImageFont.load_default()
            draw.text((x + radius + 2, y - radius), f"{name} ({visibility:.0%})", fill=color, font=font)
    
    return image, results


def main():
    print("\n" + "="*60)
    print("🐦 CUB-200 关键点检测测试")
    print("="*60 + "\n")
    
    # 设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 输出目录: {OUTPUT_DIR}")
    
    # 加载模型
    model = load_model(MODEL_PATH, device)
    
    # 获取测试图片
    test_images = get_random_images(CUB_DATASET, NUM_IMAGES)
    
    print("\n" + "-"*60)
    print("开始测试...")
    print("-"*60 + "\n")
    
    all_results = []
    
    for i, img_path in enumerate(test_images, 1):
        # 预处理
        tensor, original_image, original_size = preprocess_image(img_path, IMG_SIZE)
        tensor = tensor.to(device)
        
        # 推理
        with torch.no_grad():
            coords, vis = model(tensor)
        
        # 在原图上绘制
        result_image, keypoints = draw_keypoints(
            original_image.copy(), 
            coords[0].cpu(), 
            vis[0].cpu(),
            original_size,
            IMG_SIZE
        )
        
        # 保存结果
        filename = os.path.basename(img_path)
        output_path = os.path.join(OUTPUT_DIR, f"result_{i:02d}_{filename}")
        result_image.save(output_path)
        
        # 打印结果
        print(f"[{i}/{len(test_images)}] {filename}")
        for kp in keypoints:
            vis_str = "✅" if kp['visibility'] > 0.5 else "❌"
            print(f"    {vis_str} {kp['name']}: ({kp['x']:.1f}, {kp['y']:.1f}) - 可见度: {kp['visibility']:.0%}")
        print()
        
        all_results.append({
            'image': filename,
            'path': output_path,
            'keypoints': keypoints
        })
    
    print("="*60)
    print(f"✅ 测试完成！结果已保存到: {OUTPUT_DIR}")
    print(f"   共处理 {len(test_images)} 张图片")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    main()
