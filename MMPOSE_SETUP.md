# MMPose 测试安装指南

## 📦 安装步骤

### 1. 安装 MMPose 及其依赖

```bash
# 确保在虚拟环境中
source .venv/bin/activate

# 安装 OpenMMLab 工具链
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.0.0"

# 安装 MMPose
mim install "mmpose>=1.0.0"
```

### 2. 验证安装

```bash
python -c "import mmpose; print(mmpose.__version__)"
```

### 3. 运行测试脚本

```bash
# 直接运行（会自动查找测试图片）
python test_mmpose.py

# 或者指定图片路径
python test_mmpose.py /path/to/bird/image.jpg
```

## 🎯 测试脚本功能

`test_mmpose.py` 会自动完成以下测试：

1. ✅ **依赖检查**：验证所有必需的包是否已安装
2. 🍎 **MPS 支持检测**：检查 Apple Silicon GPU 是否可用
3. 📦 **模型列表**：显示可用的动物姿态估计模型
4. 🔍 **自动查找图片**：在常见目录查找鸟类测试照片
5. 🧪 **推理测试**：运行 MMPose 进行姿态估计
6. 📊 **结果分析**：详细解析返回的关键点数据
7. 🎨 **可视化保存**：保存标注了关键点的图片
8. 📋 **总结报告**：生成完整的测试报告

## 🦅 MMPose 能提供的信息

根据 AP-10K 数据集，MMPose 可以检测 **17 个关键点**：

1. **眼睛**：Left_Eye, Right_Eye
2. **头部**：Nose (喙部)
3. **颈部**：Neck
4. **躯干**：Root_of_tail (尾根)
5. **前肢**：
   - Left_Shoulder, Left_Elbow, Left_Front_Paw
   - Right_Shoulder, Right_Elbow, Right_Front_Paw
6. **后肢**：
   - Left_Hip, Left_Knee, Left_Back_Paw
   - Right_Hip, Right_Knee, Right_Back_Paw

### 每个关键点包含：
- **坐标**：(x, y) 像素位置
- **置信度**：0-1 的可见性分数

## 💡 可能的应用场景

1. **鸟眼检测**
   - 检测眼睛是否在画面中
   - 判断眼睛是否清晰可见
   - 计算眼部区域的锐度

2. **姿态筛选**
   - 识别飞行姿态 vs 站立姿态
   - 筛选特定角度的照片
   - 评估鸟的动作姿势

3. **完整度评估**
   - 检查身体关键部位是否可见
   - 评估遮挡程度
   - 判断照片裁切是否完整

4. **构图分析**
   - 分析头部位置（三分法）
   - 评估鸟的朝向
   - 计算身体角度

## ⚠️ 注意事项

1. **两阶段流程**：
   - 先用 YOLO 检测鸟的位置（你已有）
   - 再用 MMPose 分析关键点（新增）

2. **性能影响**：
   - MMPose 会增加处理时间
   - 建议作为**可选高级功能**
   - 可以只对 3 星照片进行姿态分析

3. **MPS 兼容性**：
   - 脚本已包含自动降级机制
   - NMS 操作会自动回退到 CPU
   - 大部分计算仍在 GPU 上进行

4. **模型下载**：
   - 首次运行会自动下载预训练模型
   - 模型大小约 100-200MB
   - 需要稳定的网络连接

## 📝 测试输出

运行成功后会生成：

1. **终端输出**：详细的测试报告和性能数据
2. **可视化图片**：`mmpose_test_result.jpg`（标注了关键点）
3. **性能指标**：初始化时间、推理时间、关键点置信度

## 🐛 可能的问题

### 问题 1：MPS 设备报错
```
RuntimeError: nms_impl: implementation for device mps:0 not found
```

**解决方案**：脚本已设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，会自动降级到 CPU

### 问题 2：模型下载失败
```
ConnectionError: Failed to download model
```

**解决方案**：检查网络连接，或手动下载模型文件

### 问题 3：内存不足
```
RuntimeError: MPS backend out of memory
```

**解决方案**：脚本会自动降级到 CPU，或减小输入图片尺寸

## 🔗 相关资源

- [MMPose 官方文档](https://mmpose.readthedocs.io/)
- [AP-10K 数据集](https://github.com/AlexTheBad/AP-10K)
- [Animal 2D Keypoint 模型库](https://mmpose.readthedocs.io/en/latest/model_zoo/animal_2d_keypoint.html)

## 📧 反馈

如有问题或建议，请记录测试结果并分享。
