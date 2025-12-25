# SuperPicky Release Notes

---

## 📦 V3.1.4 → V3.2.3 Update Summary (2025-10-25 → 2025-11-02)

### 🎯 Major Update Highlights

**From V3.1.4 to V3.2.3, SuperPicky has undergone a series of major feature upgrades and performance optimizations:**

#### ✨ Core New Features
1. **🔄 Post-Adjustment (Post-DA)** - Quickly adjust rating criteria based on existing data without re-running AI inference (V3.2.0)
2. **🌍 Full Internationalization (i18n)** - Support Chinese/English language switching with complete UI localization (V3.2.2)
3. **⚡ Significant Performance Boost** - BRISQUE calculation optimization, 3-star photos skip calculation saving 83.9% time (V3.2.2)

#### 🐛 Critical Bug Fixes
4. **Batch EXIF Writing Critical Bug** - Fixed issue that could cause metadata writing failures (V3.2.2)
5. **Post-Adjustment Interface Issues** - Fixed 4+ issues including pick flag count display, layout optimization (V3.2.1-V3.2.3)

#### 🔧 Improvements
6. **Code Signing and Packaging** - Optimized macOS code signing process, improved app security (V3.2.1)
7. **Project Configuration Enhancement** - Updated .gitignore, optimized build process (V3.2.3)

---

### 📊 Detailed Version Notes

## V3.2.3 (2025-11-02) - Project Configuration Enhancement 🔧

### 🎯 Update Focus

This is a maintenance update primarily focused on improving project configuration and code optimization.

### 🔧 Improvements

1. **Project Configuration Enhancement**
   - Updated .gitignore to exclude build and release files (*.dmg, *.log, build_*.sh)
   - Optimized project structure to avoid committing large files to repository

2. **Code Quality Improvements**
   - Optimized exiftool_manager metadata handling logic
   - Enhanced iqa_scorer quality scoring functionality
   - Improved post-adjustment dialog interaction experience

3. **Documentation and Configuration**
   - Updated application spec configuration (SuperPicky.spec)
   - Synchronized Chinese and English localization resources

### 📦 Technical Details

- Maintains all core features from V3.2.2
- Improved code maintainability and stability
- Enhanced error handling and logging output

---

## V3.2.2 (2025-10-30) - Internationalization + Performance Optimization 🌍⚡

### 🎯 Update Focus

This is an important maintenance update that completes full internationalization support and significantly optimizes BRISQUE calculation performance.

### ✨ New Features

1. **🌍 Full Internationalization (i18n) Support**
   - Support for Chinese/English interface switching
   - All UI text fully localized
   - New language files: `locales/zh_CN.json` and `locales/en_US.json`
   - Covers all text in main interface, dialogs, help documentation

### ⚡ Performance Optimization

2. **Major BRISQUE Calculation Performance Boost**
   - **Core Optimization**: 3-star (premium) photos skip BRISQUE calculation
   - **Performance Gain**: Saves **83.9%** of calculation time
   - **Logic Optimization**: Only calculates BRISQUE for 0-2 star photos (used to exclude poor quality photos)
   - **Technical Rationale**: 3-star photos have already passed dual sharpness + aesthetic validation, no need for distortion detection

3. **Help Documentation Optimization**
   - Updated BRISQUE field description explaining optimization logic
   - Enhanced Chinese and English help documentation

### 🐛 Bug Fixes

4. **Batch EXIF Metadata Writing Critical Bug**
   - Fixed issue that could cause metadata writing failures
   - Improved stability and reliability of batch operations

5. **Post-Adjustment Interface Optimization**
   - Fixed issue where pick flag count wasn't displayed
   - Increased window height for better display
   - Optimized layout and font settings

### 📦 Technical Details

- Added `LocalizationManager` class for multilingual handling
- Optimized BRISQUE calculation logic to reduce unnecessary computation
- Improved error handling for EXIF batch writing
- Maintains all core features from V3.2.0

---

## V3.2.1 (2025-10-28) - Code Signing Optimization 🔐

### 🎯 Update Focus

This is a maintenance update primarily optimizing packaging and code signing process.

### 🔧 Improvements

1. **Code Signing Optimization**
   - Optimized macOS code signing workflow
   - Enhanced application security and compatibility

2. **Packaging Process Improvement**
   - Optimized PyInstaller packaging configuration
   - Improved app launch speed and stability

### 📦 Technical Details

- Maintains all core features from V3.2.0 unchanged
- Optimized build scripts and packaging workflow
- Enhanced error handling and log output

---

## V3.2.0 (2025-10-25) - Post-Adjustment Feature 🔄

### 🎯 Major Feature Update

This is a major functional update introducing the **Post-Adjustment (Post-DA)** feature, enabling users to quickly adjust rating criteria based on existing analysis results.

### ✨ Core New Features

1. **📊 Post-Adjustment (Post-DA)**
   - Re-calculate star ratings based on existing CSV data **without re-running AI inference**
   - Real-time preview of new star distribution and change comparison
   - Drag sliders to adjust thresholds (sharpness/aesthetic/pick percentage)
   - Batch EXIF writing for quick application of new ratings
   - Speed improvement: **from 5-20 minutes down to 5-10 seconds** (60-240x faster)

2. **🎚️ Adjustable Parameters**
   - Sharpness threshold (2/3-star): 6000-9000
   - Aesthetic threshold (2/3-star): 4.5-5.5
   - Pick flag percentage: 10%-50%

3. **📈 Smart Preview**
   - Display current star distribution statistics
   - Real-time calculation of adjusted distribution
   - Comparison showing changes (+/- counts and percentages)
   - Debounce optimization (300ms) for smooth experience

### 🔧 Parameter Optimization

4. **Default Threshold Adjustments**
   - Sharpness default: 8000 → 7500 (easier to achieve premium rating)
   - Aesthetic default: 5.0 → 4.8 (easier to achieve premium rating)
   - Pick percentage: 10% → 25%, range expanded to 10-50%

### 📦 Technical Implementation

- New `PostAdjustmentEngine`: Backend rating calculation engine
- New `PostAdjustmentDialog`: Real-time preview UI dialog
- Main interface integration: Auto-detects report.csv and enables button
- Code volume: +1020 lines (714 lines core functionality)

### 🎨 User Experience

- Auto-enables "Post-Adjustment" button after processing completes
- Auto-detects historical data when browsing directories
- User-friendly 3-section layout (current stats/threshold adjustment/preview comparison)
- Complete progress display and error handling

### 📝 Use Cases

**Scenario 1: Too Few 3-Star Photos**
```
Current: Only 20 3-star photos
Action: Lower sharpness threshold 8000 → 7500
Result: 3-star increased to 35 (+15 photos)
```

**Scenario 2: Want More Picked Photos**
```
Current: Only 5 pick flags
Action: Increase pick percentage 25% → 35%
Result: Picks increased to 12 (+7 photos)
```

### 📚 Documentation

- Complete feature guide: `POST_DA_README.md`
- Unit tests: `test_post_da.py`

---

## V3.1.4 (2025-10-25) - Temp File Cleanup + Terminology Optimization

### 🎯 Update Focus

This is a maintenance update primarily improving user experience and code stability.

### ✨ New Features

1. **Complete Temp File Cleanup**
   - Reset function now completely deletes `_tmp` directory and all contents
   - Auto-cleanup of leftover `tmp_*.jpg` temporary JPEG files
   - Ensures no temp files remain after directory reset

### 🔧 Improvements

2. **BRISQUE Terminology Optimization**
   - Changed "noise" to more accurate "distortion" description
   - Better reflects actual meaning of BRISQUE technical quality scoring
   - Affects all UI displays and documentation

3. **Bug Fixes**
   - Fixed `raw_to_jpeg()` function return value issue (`None` → `False`)
   - Improved code robustness and error handling

### 📦 Technical Details

- Sharpness value format optimized (08.2f), supports values over 10000
- Removed large model files (no longer uploaded to GitHub, need local download)
- Optimized .gitignore to exclude temp_models/ directory

---

## System Requirements

- **Operating System**: macOS 10.15+ / Windows 10+ / Linux
- **Python**: 3.8+
- **Memory**: 8GB+ recommended
- **GPU**: Supported but not required (Apple Silicon MPS / NVIDIA CUDA)

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

Or run the packaged application directly.

---

## Open Source License

This software is built on the following open source projects:

- **YOLOv11**: AGPL-3.0 (Ultralytics)
- **PyIQA**: CC BY-NC-SA 4.0 (Non-commercial use)
- **ExifTool**: Perl Artistic License / GPL

**Copyright © 2024-2025 James Yu (詹姆斯·于震)**

For personal learning and non-commercial use only. Contact author for commercial licensing.

---

## Acknowledgments

Thanks to all open source project contributors and bird photographers for their feedback!

**SuperPicky - Let AI Help You Pick the Most Beautiful Moments 🦅📸**
