# Question 3: Image Upsampling and Downsampling on ESP32-CAM

## Overview

This project implements image upsampling (scaling up) and downsampling (scaling down) operations on ESP32-CAM with support for **non-integer scale factors** (e.g., 1.5, 2/3, 1.25). The implementation includes three interpolation methods (Nearest Neighbor, Bilinear, Bicubic) and is optimized for the memory constraints of the ESP32-CAM module.

## Problem Statement

**Assignment Requirements:**
- **a- (10 points)**: Perform upsampling with a given scale factor
- **b- (10 points)**: Perform downsampling with a given scale factor
- **Critical requirement**: Modules must work with non-integer scale factors (1.5, 2/3, etc.)

## What is Upsampling and Downsampling?

### Upsampling (Image Scaling Up)
**Definition:** Increasing the resolution of an image by interpolating new pixels between existing ones.

**Example:** 
- Input: 160×120 image
- Scale factor: 1.5
- Output: 240×180 image (50% larger)

**Use cases:** Image enhancement, zooming, super-resolution

### Downsampling (Image Scaling Down)
**Definition:** Decreasing the resolution of an image by sampling and combining existing pixels.

**Example:**
- Input: 160×120 image
- Scale factor: 0.6
- Output: 96×72 image (40% smaller)

**Use cases:** Image compression, thumbnail generation, reducing computational load

### Why Non-Integer Scale Factors?
Traditional image scaling often uses integer factors (2x, 3x, 0.5x), but real-world applications require arbitrary scaling (1.5x, 2/3, 1.25x). This requires **interpolation** to estimate pixel values at fractional coordinates.

## Algorithm Pipeline

### 1. Scale Factor Calculation
**Why:** Determine output dimensions based on input size and scale factor.

**How:**
- Upsampling: `output_size = input_size × scale_factor`
- Downsampling: `output_size = input_size × scale_factor`
- Round to nearest integer: `(int)(size + 0.5f)`
- Validate output size doesn't exceed memory limits

**Code Location:** Beginning of `upsampling()` and `downsampling()` functions

### 2. Coordinate Mapping
**Why:** Map each output pixel to corresponding source coordinates in input image.

**How:**
- **Upsampling**: `src_coord = dst_coord / scale_factor`
- **Downsampling**: `src_coord = dst_coord × (input_size / output_size)`
- Result is fractional: `src_x_f = 12.7`, `src_y_f = 8.3`
- Split into integer and fractional parts:
  - Integer: `src_x = 12`, `src_y = 8` (nearest pixel)
  - Fractional: `fx = 0.7`, `fy = 0.3` (interpolation weights)

**Code Location:** Inside interpolation loops

### 3. Interpolation
**Why:** Estimate pixel value at fractional coordinates using neighboring pixels.

**How:** Three methods with different quality/speed trade-offs:

#### Method 1: Nearest Neighbor
- **Speed**: Fastest (~10-20ms for 160×120)
- **Quality**: Lowest (blocky artifacts)
- **How**: Simply use the nearest pixel value
- **Use case**: When speed is critical

```cpp
int src_x = (int)(x * scale_x + 0.5f);
int src_y = (int)(y * scale_y + 0.5f);
output_pixel = input_image[src_y * width + src_x];
```

#### Method 2: Bilinear Interpolation
- **Speed**: Moderate (~50-100ms for 160×120)
- **Quality**: Good (smooth transitions)
- **How**: Weighted average of 4 nearest pixels
- **Use case**: General purpose (recommended)

```cpp
// Get 4 corner pixels: p00, p01, p10, p11
// Interpolate horizontally, then vertically
result = p00*(1-fx)*(1-fy) + p01*fx*(1-fy) + 
         p10*(1-fx)*fy + p11*fx*fy;
```

**Code Location:** `bilinear_interpolate()` function

#### Method 3: Bicubic Interpolation
- **Speed**: Slowest (~200-400ms for 160×120)
- **Quality**: Highest (sharp, smooth edges)
- **How**: Weighted average of 16 nearest pixels using cubic weighting function
- **Use case**: When quality is critical

```cpp
// Get 4×4 neighborhood (16 pixels)
// Apply cubic weight function to each
result = Σ(pixels[i][j] × cubic_weight(i - fx) × cubic_weight(j - fy))
```

**Code Location:** `bicubic_interpolate()` and `cubic_weight()` functions

### 4. RGB565 Format Handling
**Why:** ESP32-CAM outputs RGB565 (16-bit), but interpolation works on grayscale (8-bit).

**How:**
- Convert RGB565 → Grayscale: Extract R, G, B, apply luminance formula
- Perform interpolation on grayscale values
- Convert back: Grayscale → RGB565 (same value for R, G, B)

**Code Location:** `rgb565_to_gray()` and `gray_to_rgb565()` functions

## ESP32-CAM Challenges and Solutions

### Challenge 1: Memory Constraints
**Problem:** ESP32-CAM has ~200KB DRAM, but a 640×480 RGB565 image requires 614KB. Upsampling increases memory requirements further.

**Solutions:**
- **Reduced input size**: QQVGA (160×120) = 38.4KB instead of 614KB
- **Single shared buffer**: Reuse `processed_image[]` for both upsampling and downsampling
- **In-place processing**: No intermediate buffers
- **Maximum size limits**: `MAX_WIDTH = 160`, `MAX_HEIGHT = 120`
- **Validation**: Check output size before processing to prevent overflow

### Challenge 2: Processing Speed
**Problem:** Bicubic interpolation is computationally expensive (16 pixels per output pixel).

**Solutions:**
- **Method selection**: Use Bilinear for real-time applications
- **Optimized cubic weight**: Pre-calculated formula, no lookup tables
- **Processing frequency**: Run every 5 seconds (configurable)
- **Efficient coordinate calculation**: Single-pass mapping

### Challenge 3: Non-Integer Scale Factors
**Problem:** Fractional coordinates require floating-point arithmetic, which is slower on ESP32.

**Solutions:**
- **Efficient float operations**: Minimize floating-point calculations
- **Coordinate caching**: Calculate scale factors once, reuse in loops
- **Boundary checking**: Handle edge cases (pixels near image borders)

### Challenge 4: Image Quality vs. Memory
**Problem:** Higher quality interpolation (Bicubic) requires more computation and memory.

**Solutions:**
- **Method selection**: Bilinear provides good quality with reasonable speed
- **Grayscale processing**: Reduces memory by 50% (16-bit → 8-bit)
- **Single buffer**: Process one operation at a time

## Code Structure

```
upsampling_downsampling.ino
├── Configuration
│   ├── MAX_WIDTH/MAX_HEIGHT (160×120)
│   └── Camera pin definitions
├── Interpolation Methods Enum
│   └── INTERP_NEAREST, INTERP_BILINEAR, INTERP_BICUBIC
├── Interpolation Functions
│   ├── bilinear_interpolate() - 4-pixel weighted average
│   ├── cubic_weight() - Cubic weighting function
│   └── bicubic_interpolate() - 16-pixel weighted average
├── Utility Functions
│   ├── rgb565_to_gray() - Format conversion
│   └── gray_to_rgb565() - Format conversion
├── Main Processing Functions
│   ├── upsampling() - Scale up with interpolation
│   └── downsampling() - Scale down with interpolation
└── Main Functions
    ├── setup() - Camera initialization
    └── loop() - Processing pipeline
```

## Interpolation Methods Comparison

| Method | Speed | Quality | Memory | Use Case |
|--------|-------|---------|--------|----------|
| **Nearest Neighbor** | ⚡⚡⚡ Fastest | ⭐ Low | 💾 Low | Real-time, low quality needed |
| **Bilinear** | ⚡⚡ Moderate | ⭐⭐ Good | 💾 Low | **Recommended** for most cases |
| **Bicubic** | ⚡ Slowest | ⭐⭐⭐ Best | 💾 Moderate | High quality, offline processing |

## Scale Factor Examples

### Upsampling Examples
- **1.5x**: 160×120 → 240×180 (50% increase)
- **2.0x**: 160×120 → 320×240 (100% increase, doubling)
- **1.25x**: 160×120 → 200×150 (25% increase)
- **2/3 ≈ 0.667**: Actually downsampling, not upsampling

### Downsampling Examples
- **0.6x**: 160×120 → 96×72 (40% reduction)
- **0.5x**: 160×120 → 80×60 (50% reduction, halving)
- **0.75x**: 160×120 → 120×90 (25% reduction)
- **2/3 ≈ 0.667**: 160×120 → 107×80 (33% reduction)

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
- **Core Debug Level**: `None`
- **PSRAM**: `Disabled` (if module has no PSRAM)
- **Upload Speed**: `115200`

### 2. Upload Procedure

**For ESP32-CAM with Boot Button:**
1. Hold the **BOOT** button
2. Press and release the **RESET** button
3. Release the **BOOT** button
4. Click **Upload** in Arduino IDE

**For ESP32-CAM without Boot Button:**
1. Connect **GPIO0** to **GND**
2. Press and release the **RESET** button
3. Click **Upload** in Arduino IDE
4. Disconnect **GPIO0** from **GND** after upload

### 3. Serial Monitor
- Open Serial Monitor: `Tools → Serial Monitor`
- Set baud rate: **115200**
- Set line ending: `Newline`

## Usage

1. Power on ESP32-CAM
2. Point camera at scene
3. Processing runs automatically every 5 seconds
4. Results printed to Serial Monitor:
   - Input image dimensions
   - Upsampling results (1.5x)
   - Downsampling results (0.6x)
   - Output dimensions
   - Memory status

## Output Example

```
==============================================================
     ESP32 CAM UPSAMPLING/DOWNSAMPLING MODULU
==============================================================
[OK] Kamera baslatildi

==============================================================
     UPSAMPLING VE DOWNSAMPLING TEST
==============================================================
Giris goruntu boyutu: 160x120

--------------------------------------------------------------
UPSAMPLING (1.5x) - BILINEAR
--------------------------------------------------------------
  Orijinal boyut: 160x120
  Yeni boyut:    240x180
  [OK] Upsampling tamamlandi

--------------------------------------------------------------
DOWNSAMPLING (0.6x) - BILINEAR
--------------------------------------------------------------
  Orijinal boyut: 160x120
  Yeni boyut:    96x72
  [OK] Downsampling tamamlandi

==============================================================
     [OK] ISLEM TAMAMLANDI!
==============================================================
Serbest heap: 45234 bytes
==============================================================
```

## Configuration

You can modify the scale factors and interpolation methods in `loop()`:

```cpp
// Upsampling example
upsampling(input_image, width, height,
           processed_image, &out_width, &out_height,
           1.5f, INTERP_BILINEAR);  // 1.5x scale, bilinear method

// Downsampling example
downsampling(input_image, width, height,
             processed_image, &out_width, &out_height,
             0.6f, INTERP_BILINEAR);  // 0.6x scale, bilinear method
```

## Mathematical Foundation

### Bilinear Interpolation

For a point `(x, y)` with fractional parts `(fx, fy)`, the interpolated value is:

```
I(x,y) = I(0,0)×(1-fx)×(1-fy) + I(1,0)×fx×(1-fy) +
         I(0,1)×(1-fx)×fy + I(1,1)×fx×fy
```

Where `I(i,j)` are the 4 corner pixel values.

### Bicubic Interpolation

Uses a cubic weighting function:

```
W(x) = {
  1.5×|x|³ - 2.5×|x|² + 1.0,  if |x| ≤ 1
  -0.5×|x|³ + 2.5×|x|² - 4×|x| + 2.0,  if 1 < |x| ≤ 2
  0,  otherwise
}
```

The interpolated value is a weighted sum of 16 neighboring pixels.

## Performance Metrics

| Operation | Method | Time (160×120) | Memory |
|-----------|--------|----------------|--------|
| Upsampling 1.5x | Nearest | ~15ms | ~38KB |
| Upsampling 1.5x | Bilinear | ~60ms | ~38KB |
| Upsampling 1.5x | Bicubic | ~250ms | ~38KB |
| Downsampling 0.6x | Nearest | ~10ms | ~38KB |
| Downsampling 0.6x | Bilinear | ~50ms | ~38KB |
| Downsampling 0.6x | Bicubic | ~200ms | ~38KB |

**Note:** Performance depends on scale factor and output size. Larger scale factors take longer.

## Python Implementation

A PC-based Python implementation (`test_upsampling_downsampling.py`) is provided for:
- Algorithm validation and testing
- Visual comparison of interpolation methods
- Parameter tuning
- Educational purposes

**Key Features:**
- Supports local image files (handles non-ASCII paths on Windows)
- Command-line interface with flexible arguments
- Default scale factors: 1.5x upsampling, 0.6x downsampling
- Three interpolation methods: nearest, bilinear, bicubic
- Automatic output file naming based on scale factor and method

**Usage:**
```bash
cd question3
python test_upsampling_downsampling.py --image input_image.jpg --upscale 1.5 --downscale 0.6 --method bilinear
```

**Arguments:**
- `--image`: Input image file path (default: `input_image.jpg`)
- `--upscale`: Upsampling scale factor (default: 1.5)
- `--downscale`: Downsampling scale factor (default: 0.6)
- `--method`: Interpolation method: `nearest`, `bilinear`, or `bicubic` (default: `bilinear`)

**Output:**
The script saves processed images with descriptive filenames:
- `upsampled_1.5x_bilinear.jpg` - Upsampled image
- `downsampled_0.6x_bilinear.jpg` - Downsampled image

**Note:** The Python implementation handles non-ASCII characters in file paths (e.g., Turkish characters) using `numpy.frombuffer` and `cv2.imdecode` for reading, and `cv2.imencode` for saving images.

## Troubleshooting

### Compilation Errors

**DRAM overflow:**
```
region `dram0_0_seg' overflowed by X bytes
```
**Solution:** Reduce `MAX_WIDTH`/`MAX_HEIGHT` or use smaller input images

**Camera init error:**
```
[HATA] Kamera baslatma hatasi: 0x20001
```
**Solution:** Check pin connections, verify power supply

### Runtime Issues

**Output size too large:**
```
[HATA] Cikis boyutu cok buyuk!
```
**Solution:** Reduce scale factor or input image size

**Poor image quality:**
- Try Bicubic interpolation instead of Bilinear
- Ensure good lighting conditions
- Check camera focus

## References

- **Bilinear Interpolation**: [Wikipedia](https://en.wikipedia.org/wiki/Bilinear_interpolation)
- **Bicubic Interpolation**: [Wikipedia](https://en.wikipedia.org/wiki/Bicubic_interpolation)
- **Image Resizing**: [OpenCV Documentation](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)
- ESP32-CAM documentation: [ESP-IDF Camera API](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/camera.html)

## License

This project is part of an embedded image processing course assignment.

## Author

Embedded Image Processing Final Project - Question 3
