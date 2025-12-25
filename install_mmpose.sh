#!/bin/bash
# MMPose 安装脚本 - 适用于 macOS (Apple Silicon)

set -e  # 遇到错误立即退出

echo "=========================================="
echo "🦅 MMPose 安装脚本"
echo "=========================================="

# 检查是否在虚拟环境中
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  未检测到虚拟环境"
    echo "建议先激活虚拟环境: source .venv/bin/activate"
    read -p "是否继续安装到全局环境? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 安装已取消"
        exit 1
    fi
fi

# 显示 Python 版本
echo ""
echo "🐍 Python 版本:"
python --version

# 显示 PyTorch 版本
echo ""
echo "🔥 PyTorch 版本:"
python -c "import torch; print(f'PyTorch {torch.__version__}')" || echo "⚠️  PyTorch 未安装"

# 检查 MPS 支持
echo ""
echo "🍎 检查 Apple Silicon MPS 支持:"
python -c "import torch; print(f'MPS 可用: {torch.backends.mps.is_available()}')" || echo "⚠️  无法检查 MPS"

echo ""
echo "=========================================="
echo "📦 开始安装 MMPose 及依赖..."
echo "=========================================="

# 安装 OpenMMLab 的包管理工具 MIM
echo ""
echo "1️⃣ 安装 OpenMIM..."
pip install -U openmim

# 安装 MMEngine (OpenMMLab 的核心引擎)
echo ""
echo "2️⃣ 安装 MMEngine..."
mim install mmengine

# 安装 MMCV (计算机视觉基础库)
echo ""
echo "3️⃣ 安装 MMCV (可能需要几分钟)..."
mim install "mmcv>=2.0.1"

# 安装 MMDetection (目标检测，MMPose 的依赖)
echo ""
echo "4️⃣ 安装 MMDetection..."
mim install "mmdet>=3.0.0"

# 安装 MMPose (姿态估计)
echo ""
echo "5️⃣ 安装 MMPose..."
mim install "mmpose>=1.0.0"

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="

# 验证安装
echo ""
echo "🔍 验证安装..."
python -c "
import sys
try:
    import mmpose
    print(f'✅ MMPose: v{mmpose.__version__}')
except ImportError as e:
    print(f'❌ MMPose: {e}')
    sys.exit(1)

try:
    import mmcv
    print(f'✅ MMCV: v{mmcv.__version__}')
except ImportError as e:
    print(f'❌ MMCV: {e}')
    sys.exit(1)

try:
    import mmengine
    print(f'✅ MMEngine: v{mmengine.__version__}')
except ImportError as e:
    print(f'❌ MMEngine: {e}')
    sys.exit(1)

try:
    import mmdet
    print(f'✅ MMDetection: v{mmdet.__version__}')
except ImportError as e:
    print(f'❌ MMDetection: {e}')
    sys.exit(1)

print('')
print('🎉 所有依赖安装成功！')
"

echo ""
echo "=========================================="
echo "🚀 现在可以运行测试脚本:"
echo "   python test_mmpose.py"
echo "=========================================="
