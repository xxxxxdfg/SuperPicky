#!/usr/bin/env python3
"""
头部区域锐度测试脚本
使用眼睛为圆心，眼喙距离×1.2为半径，与seg掩码取交集
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO
from find_bird_util import raw_to_jpeg

# 配置
MODEL_PATH = "models/cub200_keypoint_resnet50.pth"
YOLO_MODEL_PATH = "models/yolo11m-seg.pt"
IMG_SIZE = 416
OUTPUT_DIR = "/Users/jameszhenyu/Desktop/head_region_test"
RADIUS_MULTIPLIER = 1.2
NO_BEAK_RADIUS_RATIO = 0.15  # 无喙时用检测框的15%


class PartLocalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.coord_head = nn.Linear(256, 6)
        self.vis_head = nn.Linear(256, 3)

    def forward(self, x):
        features = self.head(self.backbone(x))
        coords = torch.sigmoid(self.coord_head(features)).view(-1, 3, 2)
        vis = torch.sigmoid(self.vis_head(features))
        return coords, vis


def calculate_distance(p1, p2):
    """计算两点距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def select_eye_for_center(left_eye, right_eye, beak, left_vis, right_vis):
    """
    选择用哪只眼睛作为圆心
    规则：用更远离鸟喙的那只眼睛
    """
    if left_vis >= 0.5 and right_vis >= 0.5:
        # 两眼都可见，选更远离喙的
        left_dist = calculate_distance(left_eye, beak)
        right_dist = calculate_distance(right_eye, beak)
        if left_dist >= right_dist:
            return left_eye, 'left'
        else:
            return right_eye, 'right'
    elif left_vis >= 0.5:
        return left_eye, 'left'
    elif right_vis >= 0.5:
        return right_eye, 'right'
    else:
        return None, None


def create_head_mask(image_shape, eye_center, radius, seg_mask):
    """
    创建头部区域掩码
    = 以眼睛为圆心的圆 ∩ 鸟体seg掩码
    """
    h, w = image_shape[:2]
    
    # 创建圆形掩码
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (int(eye_center[0]), int(eye_center[1])), int(radius), 255, -1)
    
    # 与seg掩码取交集
    head_mask = cv2.bitwise_and(circle_mask, seg_mask)
    
    return head_mask, circle_mask


def calculate_sharpness(image, mask):
    """计算掩码区域的锐度"""
    if mask.sum() == 0:
        return 0.0
    
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 只在掩码区域计算
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # 计算Laplacian
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    
    # 只取掩码内的方差
    mask_pixels = mask > 0
    if mask_pixels.sum() == 0:
        return 0.0
    
    sharpness = laplacian[mask_pixels].var()
    return sharpness


def visualize_result(image, seg_mask, head_mask, circle_mask, eye_center, beak, radius, sharpness, filename, beak_visible=True):
    """可视化结果"""
    vis_img = image.copy()
    h, w = image.shape[:2]
    
    # 1. 绘制seg掩码轮廓（绿色）
    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
    
    # 2. 绘制头部圆（黄色虚线）
    cv2.circle(vis_img, (int(eye_center[0]), int(eye_center[1])), int(radius), (255, 255, 0), 2)
    
    # 3. 头部区域填充蓝色半透明
    overlay = vis_img.copy()
    overlay[head_mask > 0] = [100, 150, 255]  # 蓝色
    vis_img = cv2.addWeighted(overlay, 0.5, vis_img, 0.5, 0)
    
    # 4. 标记眼睛（红色圆点）
    cv2.circle(vis_img, (int(eye_center[0]), int(eye_center[1])), 5, (255, 0, 0), -1)
    
    # 5. 标记喙（绿色圆点，如果可见）
    if beak_visible:
        cv2.circle(vis_img, (int(beak[0]), int(beak[1])), 5, (0, 255, 0), -1)
    
    # 6. 添加文字
    cv2.putText(vis_img, f"Sharpness: {sharpness:.0f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Radius: {radius:.0f}px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 7. 喙可见性提示
    if not beak_visible:
        cv2.putText(vis_img, "NO BEAK (15% fallback)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)  # 橙色
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, f"head_{filename}")
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    return output_path


def process_image(jpg_path, yolo_model, keypoint_model, device, transform):
    """处理单张图片"""
    filename = os.path.basename(jpg_path)
    
    # YOLO 检测
    yolo_results = yolo_model(jpg_path, verbose=False)
    
    bird_crop = None
    seg_mask = None
    box = None
    
    for r in yolo_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for i, cls in enumerate(r.boxes.cls):
                if int(cls) == 14:  # bird
                    box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    
                    # 获取seg掩码
                    if r.masks is not None and i < len(r.masks):
                        mask = r.masks[i].data.cpu().numpy()[0]
                        seg_mask = (mask * 255).astype(np.uint8)
                        # resize到原图大小
                        img = cv2.imread(jpg_path)
                        seg_mask = cv2.resize(seg_mask, (img.shape[1], img.shape[0]))
                    
                    img = cv2.imread(jpg_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bird_crop = img[box[1]:box[3], box[0]:box[2]]
                    break
        if bird_crop is not None:
            break
    
    if bird_crop is None or seg_mask is None:
        return None
    
    # 关键点检测（在裁剪区域上）
    pil_crop = Image.fromarray(bird_crop)
    tensor = transform(pil_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        coords, vis = keypoint_model(tensor)
    
    coords = coords[0].cpu().numpy()
    vis = vis[0].cpu().numpy()
    
    # 坐标转换到原图
    crop_h, crop_w = bird_crop.shape[:2]
    left_eye = (coords[0, 0] * crop_w + box[0], coords[0, 1] * crop_h + box[1])
    right_eye = (coords[1, 0] * crop_w + box[0], coords[1, 1] * crop_h + box[1])
    beak = (coords[2, 0] * crop_w + box[0], coords[2, 1] * crop_h + box[1])
    
    # 选择眼睛
    eye_center, eye_name = select_eye_for_center(left_eye, right_eye, beak, vis[0], vis[1])
    
    if eye_center is None:
        return {'filename': filename, 'status': 'no_visible_eye'}
    
    # 计算半径
    beak_visible = vis[2] >= 0.5
    if beak_visible:  # 喙可见
        radius = calculate_distance(eye_center, beak) * RADIUS_MULTIPLIER
    else:
        # 喙不可见，用检测框的15%
        radius = max(box[2] - box[0], box[3] - box[1]) * NO_BEAK_RADIUS_RATIO
    
    # 创建头部掩码
    img = cv2.imread(jpg_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    head_mask, circle_mask = create_head_mask(img.shape, eye_center, radius, seg_mask)
    
    # 计算锐度
    sharpness = calculate_sharpness(img, head_mask)
    
    # 可视化
    vis_path = visualize_result(img, seg_mask, head_mask, circle_mask, 
                                 eye_center, beak, radius, sharpness, filename, beak_visible)
    
    return {
        'filename': filename,
        'status': 'ok',
        'eye_used': eye_name,
        'radius': radius,
        'sharpness': sharpness,
        'beak_visible': beak_visible,
        'visualization': vis_path
    }


def main():
    print("\n" + "="*60)
    print("🐦 头部区域锐度测试")
    print("="*60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    print("加载关键点模型...")
    keypoint_model = PartLocalizer()
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    keypoint_model.load_state_dict(checkpoint['model_state_dict'])
    keypoint_model.to(device).eval()
    
    print("加载YOLO模型...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试全部目录
    test_dir = "/Users/jameszhenyu/Desktop/2025-10-18/3星_优选"
    raw_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.nef') and not os.path.isdir(os.path.join(test_dir, f))]
    
    # 构建完整路径
    raw_paths = [os.path.join(test_dir, f) for f in raw_files]
    
    print(f"\\n测试 {len(raw_paths)} 张图片...")
    print("-"*60)
    
    results = []
    for i, raw_path in enumerate(raw_paths, 1):
        jpg_path = os.path.splitext(raw_path)[0] + '.jpg'
        
        # 转换 RAW 到 JPG
        raw_to_jpeg(raw_path)
        
        if not os.path.exists(jpg_path):
            print(f"[{i}] {os.path.basename(raw_path)} - 转换失败")
            continue
        
        try:
            result = process_image(jpg_path, yolo_model, keypoint_model, device, transform)
            if result and result['status'] == 'ok':
                beak_status = "✓" if result['beak_visible'] else "✗无喙"
                print(f"[{i}] {result['filename']}")
                print(f"    眼睛: {result['eye_used']}, 半径: {result['radius']:.0f}px, 锐度: {result['sharpness']:.0f}, 喙: {beak_status}")
                results.append(result)
            elif result:
                print(f"[{i}] {os.path.basename(raw_path)} - {result['status']}")
        except Exception as e:
            print(f"[{i}] {os.path.basename(raw_path)} - 错误: {e}")
        finally:
            if os.path.exists(jpg_path):
                os.remove(jpg_path)
    
    print("\n" + "="*60)
    print(f"✅ 完成！可视化结果已保存到: {OUTPUT_DIR}")
    print(f"   共处理 {len(results)} 张图片")
    print("="*60)


if __name__ == "__main__":
    main()
