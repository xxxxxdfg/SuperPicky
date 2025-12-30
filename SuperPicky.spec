# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

# 获取当前工作目录
base_path = os.path.abspath('.')

# 动态收集数据文件
ultralytics_datas = collect_data_files('ultralytics')
pyiqa_datas = collect_data_files('pyiqa')

# 组合所有数据文件
all_datas = [
    # AI模型文件
    (os.path.join(base_path, 'models/yolo11m-seg.pt'), 'models'),
    # V3.5 新增：鸟类关键点检测模型
    (os.path.join(base_path, 'models/cub200_keypoint_resnet50.pth'), 'models'),
    # V3.5 新增：飞行姿态检测模型
    (os.path.join(base_path, 'models/superFlier_efficientnet.pth'), 'models'),

    # ExifTool 完整打包
    (os.path.join(base_path, 'exiftool_bundle'), 'exiftool_bundle'),

    # 图片资源
    (os.path.join(base_path, 'img'), 'img'),

    # 国际化语言包
    (os.path.join(base_path, 'locales'), 'locales'),
]
# 添加动态收集的数据
all_datas.extend(ultralytics_datas)
all_datas.extend(pyiqa_datas)

a = Analysis(
    ['main.py'],
    pathex=[base_path],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        'ultralytics',
        'torch',
        'torchvision',
        'PIL',
        'PIL._tkinter_finder',
        'tkinter',
        'tkinter.ttk',
        'cv2',
        'numpy',
        'yaml',
        'ttkthemes',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        # PyIQA 隐藏导入（修复 FileNotFoundError）
        'pyiqa',
        'pyiqa.models',
        'pyiqa.archs',
        'pyiqa.data',
        'pyiqa.utils',
        'pyiqa.metrics',
        'pyiqa.losses',
        'pyiqa.matlab_utils',
        # PyIQA 依赖库（V3.2.1新增）
        'scipy',
        'scipy.stats',
        'scipy.special',
        'scipy.optimize',
        'scipy.linalg',
        'scipy.io',
        'timm',
        'timm.models',
        'timm.models.layers',
        'einops',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SuperPicky',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity='Developer ID Application: James Zhen Yu (JWR6FDB52H)',
    entitlements_file='entitlements.plist',
    icon='img/SuperPicky-V0.02.icns',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SuperPicky',
)

app = BUNDLE(
    coll,
    name='SuperPicky.app',
    icon='img/SuperPicky-V0.02.icns',
    bundle_identifier='com.jamesphotography.superpicky',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleName': 'SuperPicky',
        'CFBundleDisplayName': 'SuperPicky - 慧眼选鸟',
        'CFBundleVersion': '3.5.0',
        'CFBundleShortVersionString': '3.5.0',
        'NSHumanReadableCopyright': 'Copyright © 2025 James Zhen Yu. All rights reserved.',
        'LSMinimumSystemVersion': '10.15',
        'NSRequiresAquaSystemAppearance': False,
    },
)
