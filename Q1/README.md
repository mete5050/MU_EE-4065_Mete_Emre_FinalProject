# Question 1: Object Detection using Thresholding on ESP32-CAM

## Overview

This project implements an optimized real-time object detection system on ESP32-CAM using advanced thresholding techniques. The system detects bright objects of approximately **1000 pixels** (±50 tolerance) in images where object pixels are significantly brighter than background pixels. The implementation is specifically optimized for the severe memory constraints of the ESP32-CAM module.

## Problem Statement

**Assignment Requirements:**
- Detect a bright object with approximately 1000 pixels in an image
- Background pixels are darker compared to object pixels
- Extract the object based on its size using thresholding
- Implement complete solution on ESP32-CAM

## Algorithm Pipeline

The detection pipeline consists of 7 sequential stages, each optimized for memory efficiency:

### 1. RGB565 to Grayscale Conversion
**Why:** ESP32-CAM outputs images in RGB565 format (16-bit per pixel), but thresholding algorithms work on single-channel grayscale images (8-bit per pixel).

**How:** 
- Extract R, G, B components from 16-bit RGB565 pixel
- Apply standard luminance formula: `Y = 0.299*R + 0.587*G + 0.114*B`
- This conversion reduces memory usage by 50% (16-bit → 8-bit)

**Code Location:** `rgb565_to_gray()` function

### 2. Noise Reduction (In-Place Filtering)
**Why:** Camera sensors introduce noise that can create false positives in thresholding. Traditional Gaussian blur requires storing the entire filtered image, which is impossible on ESP32-CAM.

**How:**
- Implemented 5-point cross filter (up, left, center, right, down)
- **In-place processing**: Filter directly on the input buffer
- Uses only a single row buffer (`temp_line[IMAGE_WIDTH]`) instead of full image copy
- Weighted average: center pixel has weight 2, neighbors have weight 1
- This saves ~19KB of DRAM compared to traditional approaches

**Code Location:** `apply_noise_reduction()` function

### 3. Otsu Threshold Calculation
**Why:** Manual threshold selection is unreliable under varying lighting conditions. Otsu's method automatically finds the optimal threshold by maximizing inter-class variance.

**How:**
- Build histogram of grayscale values (0-255)
- For each possible threshold (0-255):
  - Calculate class probabilities: `q1` (foreground), `q2` (background)
  - Calculate class means: `μ1` (foreground), `μ2` (background)
  - Calculate inter-class variance: `σ² = q1 × q2 × (μ1 - μ2)²`
- Select threshold with maximum variance
- **Adjustment for bright objects**: If calculated threshold is too low (< 50), adjust upward to better separate bright objects from dark backgrounds

**Code Location:** `calculate_otsu_threshold()` function

### 4. Binary Thresholding
**Why:** Convert grayscale image to binary (black/white) to clearly separate object from background.

**How:**
- For each pixel: if value > threshold → white (255), else → black (0)
- Simple comparison operation, very memory-efficient

**Code Location:** `apply_threshold()` function

### 5. Small Noise Removal
**Why:** Binary thresholding can leave isolated noise pixels that create false blobs.

**How:**
- Check 4-connected neighbors (up, down, left, right) for each white pixel
- If a white pixel has fewer than 2 white neighbors, remove it (set to black)
- **In-place operation**: Works directly on binary image
- This removes isolated pixels while preserving connected object regions

**Code Location:** `remove_small_noise()` function

### 6. Connected Components Labeling
**Why:** Identify distinct blobs (connected regions) in the binary image to analyze each separately.

**How:**
- **Two-pass algorithm** with union-find data structure:
  - **First pass**: Scan image, assign labels, record equivalences when neighbors have different labels
  - **Second pass**: Resolve equivalences using union-find with path compression
- **4-connectivity**: Only checks horizontal and vertical neighbors (not diagonal)
- Uses `uint16_t` labels array to minimize memory (instead of `int`)
- Maximum labels limited to 100 to prevent memory overflow

**Code Location:** `label_connected_components()` function

### 7. Blob Analysis and Object Detection
**Why:** Calculate properties of each blob and filter by target size (1000 ± 50 pixels).

**How:**
- For each labeled component:
  - Calculate area (pixel count)
  - Calculate centroid: `(sum_x / area, sum_y / area)`
  - Calculate bounding box: `(min_x, min_y)` to `(max_x, max_y)`
- Filter blobs: `abs(area - TARGET_OBJECT_SIZE) <= SIZE_TOLERANCE`
- Return detected objects with their properties

**Code Location:** `calculate_blob_properties()` and `detect_object_by_size()` functions

## ESP32-CAM Challenges and Solutions

### Challenge 1: Severe Memory Constraints
**Problem:** ESP32-CAM has only ~200KB of DRAM, but a 640×480 RGB565 image requires 614KB.

**Solutions:**
- Reduced image size to **QQVGA (160×120)**: 38.4KB instead of 614KB (94% reduction)
- Used **in-place processing**: No intermediate image buffers
- Changed data types: `uint16_t` for labels, `uint8_t` for coordinates
- Limited `MAX_LABELS` to 100 (from 1000)
- Single frame buffer: `fb_count = 1`
- Static buffers reused across function calls

### Challenge 2: Stack Overflow
**Problem:** Large arrays on stack cause crashes.

**Solutions:**
- Made all large buffers `static`: `static uint8_t gray_image[...]`, `static uint16_t labels[...]`
- Static buffers are allocated in BSS segment, not stack
- Single row buffer for noise reduction instead of full image copy

### Challenge 3: Processing Speed
**Problem:** Complex algorithms must run in real-time.

**Solutions:**
- Optimized Otsu calculation: single-pass histogram building
- Efficient union-find with path compression
- Processing frequency set to 1 FPS (configurable)
- Removed unnecessary operations (e.g., full Gaussian blur)

### Challenge 4: Camera Format Limitations
**Problem:** ESP32-CAM outputs RGB565, but we need grayscale.

**Solutions:**
- Efficient RGB565→grayscale conversion using bit shifts
- Process directly from camera buffer without copying
- No format conversion overhead

## Code Structure

```
esp32_thresholding_arduino.ino
├── Configuration Section
│   ├── TARGET_OBJECT_SIZE (1000 pixels)
│   ├── SIZE_TOLERANCE (±50 pixels)
│   ├── IMAGE_WIDTH/HEIGHT (160×120)
│   └── MAX_LABELS (100)
├── Data Structures
│   └── blob_t (area, centroid, bounding box)
├── Camera Pin Definitions
│   └── ESP32-CAM AI-Thinker pin mapping
├── Image Processing Functions
│   ├── rgb565_to_gray() - Format conversion
│   ├── apply_noise_reduction() - In-place filtering
│   ├── calculate_otsu_threshold() - Automatic threshold
│   ├── apply_threshold() - Binary conversion
│   ├── remove_small_noise() - Morphological cleanup
│   ├── label_connected_components() - Blob detection
│   ├── calculate_blob_properties() - Feature extraction
│   └── detect_object_by_size() - Size-based filtering
└── Main Functions
    ├── setup() - Camera initialization
    └── loop() - Processing pipeline
```

## Hardware Requirements

- **ESP32-CAM** (AI-Thinker module with OV2640 sensor)
- USB-to-Serial adapter (FTDI, CP2102, or similar)
- Power supply: 5V, 2A (stable power is critical for camera)
- Optional: Breadboard and jumper wires for GPIO0 connection

## Software Requirements

- **Arduino IDE** (1.8.x or 2.x)
- **ESP32 Board Support Package**
  - Install via: `File → Preferences → Additional Board Manager URLs`
  - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
  - Install: `Tools → Board Manager → esp32`

## Installation and Upload

### 1. Board Configuration
- **Board**: `Tools → Board → ESP32 Arduino → AI Thinker ESP32-CAM`
- **Partition Scheme**: `No OTA (2MB APP/2MB SPIFFS)`
- **Core Debug Level**: `None` (reduces memory usage)
- **PSRAM**: `Disabled` (if module has no PSRAM)
- **Upload Speed**: `115200`

### 2. Upload Procedure

**For ESP32-CAM with Boot Button:**
1. Hold the **BOOT** button
2. Press and release the **RESET** button
3. Release the **BOOT** button
4. Click **Upload** in Arduino IDE
5. Wait for "Connecting..." message, then release if needed

**For ESP32-CAM without Boot Button:**
1. Connect **GPIO0** to **GND** using jumper wire
2. Press and release the **RESET** button
3. Click **Upload** in Arduino IDE
4. Disconnect **GPIO0** from **GND** after upload completes

### 3. Serial Monitor
- Open Serial Monitor: `Tools → Serial Monitor`
- Set baud rate: **115200**
- Set line ending: `Newline` or `Both NL & CR`

## Usage

1. Power on ESP32-CAM (ensure stable 5V, 2A supply)
2. Point camera at scene with bright object
3. Detection runs automatically every 1 second
4. Results printed to Serial Monitor:
   - Calculated threshold value
   - Number of components detected
   - Object detection status
   - Blob properties (if detected)

## Output Example

```
==============================================================
     ESP32-CAM THRESHOLDING OBJECT DETECTION
==============================================================
[OK] Kamera baslatildi

==============================================================
     PROCESSING PIPELINE
==============================================================
  1. RGB565 -> Grayscale conversion
  2. Noise reduction (in-place)
  3. Otsu threshold calculation
  4. Binary thresholding
  5. Small noise removal
  6. Connected components labeling
  7. Blob analysis
==============================================================

Calculated threshold: 145
Components detected: 3

=== OBJECT DETECTED ===
Size: 1002 pixels
Center: (80, 60)
Bounding Box: (50, 30) - (110, 90)
========================

Serbest heap: 45234 bytes
==============================================================
```

## Configuration Parameters

You can adjust these parameters in the code:

```cpp
#define TARGET_OBJECT_SIZE 1000  // Target object size in pixels
#define SIZE_TOLERANCE 50        // Size tolerance (±pixels)
#define IMAGE_WIDTH 160         // Image width (QQVGA)
#define IMAGE_HEIGHT 120        // Image height (QQVGA)
#define MAX_LABELS 100          // Maximum number of labels
```

**Note:** Increasing `IMAGE_WIDTH`/`IMAGE_HEIGHT` or `MAX_LABELS` may cause DRAM overflow. Current values are optimized for ESP32-CAM memory constraints.

## Performance Metrics

- **Processing time**: ~200-300ms per frame
- **Frame rate**: 1 FPS (configurable in `loop()`)
- **Memory usage**: 
  - DRAM: ~40KB (static buffers)
  - Flash: ~50KB (code)
- **Accuracy**: Depends on:
  - Lighting conditions
  - Object-background contrast
  - Object size (must be close to 1000 pixels)

## Algorithm Details

### Otsu Thresholding Mathematical Foundation

Otsu's method maximizes inter-class variance:

```
σ² = q1 × q2 × (μ1 - μ2)²
```

Where:
- `q1`, `q2`: Class probabilities (foreground, background)
- `μ1`, `μ2`: Class means (foreground, background)

The algorithm tests all possible thresholds (0-255) and selects the one with maximum variance. This ensures optimal separation between object and background.

### Connected Components Labeling

**Two-pass algorithm:**
1. **First pass**: Scan image left-to-right, top-to-bottom
   - Assign labels to white pixels
   - Record equivalences when neighbors have different labels
2. **Second pass**: Resolve equivalences using union-find
   - Path compression for efficiency
   - Update all pixels with final labels

**4-connectivity**: Only checks horizontal and vertical neighbors (not diagonal), which is more memory-efficient than 8-connectivity.

## Troubleshooting

### Compilation Errors

**DRAM overflow:**
```
region `dram0_0_seg' overflowed by X bytes
```
**Solution:** Reduce `MAX_LABELS`, `IMAGE_WIDTH`, or `IMAGE_HEIGHT`

**Camera init error:**
```
[HATA] Kamera baslatma hatasi: 0x20001
```
**Solution:** Check pin connections, verify power supply (5V, 2A)

**Upload fails:**
```
Failed to connect to ESP32: Wrong boot mode detected
```
**Solution:** Ensure ESP32-CAM is in boot mode (GPIO0 to GND or boot button)

### Runtime Issues

**No detection:**
- Adjust `TARGET_OBJECT_SIZE` and `SIZE_TOLERANCE`
- Check lighting conditions
- Verify object is bright enough
- Check Serial Monitor for threshold value

**False positives:**
- Increase noise reduction intensity
- Adjust Otsu threshold adjustment factor
- Reduce `SIZE_TOLERANCE`

**Camera not working:**
- Verify power supply (needs stable 5V, 2A)
- Check all pin connections
- Try different USB-to-Serial adapter

## Python Implementation

A PC-based Python implementation (`python_thresholding.py`) is also provided for:
- Algorithm validation and testing
- Real-time camera detection on PC
- Parameter tuning and visualization
- Educational purposes

See `PYTHON_USAGE.md` for details.

## References

- Otsu, N. (1979). "A threshold selection method from gray-level histograms". *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62-66.
- Connected Components Labeling algorithms
- ESP32-CAM documentation: [ESP-IDF Camera API](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/camera.html)

## License

This project is part of an embedded image processing course assignment.

## Author

Embedded Image Processing Final Project - Question 1
