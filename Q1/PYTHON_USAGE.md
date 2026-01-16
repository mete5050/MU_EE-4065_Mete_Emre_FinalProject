# Python Thresholding Usage Guide

## Overview

The Python implementation (`python_thresholding.py`) provides a PC-based version of the ESP32-CAM thresholding algorithm. It uses the same processing pipeline for validation and testing purposes.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install opencv-python numpy scipy scikit-image matplotlib
```

## Usage

### Real-time Camera Detection (Recommended)

Open webcam and detect objects in real-time:
```bash
python python_thresholding.py --camera
```

Or specify camera ID:
```bash
python python_thresholding.py --camera 0
```

### Basic Image Processing

Process an image with default settings:
```bash
python python_thresholding.py --image input.jpg
```

### Command Line Options

```bash
python python_thresholding.py [OPTIONS]

Options:
  -c, --camera [ID]         Use webcam for real-time detection (camera ID, default: 0)
  -i, --image PATH          Input image path
  -o, --output PATH         Output image path (default: result_thresholding.jpg)
  -s, --target-size SIZE    Target object size in pixels (default: 1000)
  -t, --threshold VALUE     Manual threshold value (None for Otsu)
  --show-pipeline           Show processing pipeline visualization
  --create-test             Create and process test image
```

### Examples

1. **Real-time detection from webcam:**
```bash
python python_thresholding.py --camera
```

2. **Real-time with custom target size:**
```bash
python python_thresholding.py --camera -s 1500
```

3. **Process an image and save result:**
```bash
python python_thresholding.py -i esp32_cam_image.jpg -o result.jpg
```

4. **Create test image and process it:**
```bash
python python_thresholding.py --create-test --show-pipeline
```

5. **Use custom target size:**
```bash
python python_thresholding.py -i image.jpg -s 1500
```

6. **Use manual threshold:**
```bash
python python_thresholding.py -i image.jpg -t 128
```

7. **Show processing pipeline:**
```bash
python python_thresholding.py -i image.jpg --show-pipeline
```

## Real-time Camera Controls

When using `--camera` mode, you can use the following keyboard controls:

- **'q'** - Quit the program
- **'s'** - Save current frame to file
- **'r'** - Reset threshold to Otsu (automatic)
- **'+' or '='** - Increase threshold by 5
- **'-' or '_'** - Decrease threshold by 5
- **Ctrl+C** - Exit gracefully

## Processing Pipeline

The script follows the same 8-step pipeline as the ESP32 code:

1. **RGB to Grayscale** - Convert color image to grayscale
2. **Noise Reduction** - Apply Gaussian blur (3x3 kernel)
3. **Otsu Threshold** - Calculate optimal threshold automatically
4. **Binary Thresholding** - Convert to black/white image
5. **Noise Removal** - Morphological opening to remove small noise
6. **Connected Components** - Label all distinct blobs
7. **Blob Analysis** - Calculate properties (area, centroid, bbox)
8. **Object Detection** - Find objects matching target size

## Output

The script generates:

1. **Result Image** (`result_thresholding.jpg`)
   - Original image with detected objects marked
   - Green bounding boxes around detected objects
   - Red dots at centroids
   - Area labels

2. **Binary Image** (`result_thresholding_binary.jpg`)
   - Binary thresholded image after noise removal

3. **Pipeline Visualization** (`result_thresholding_pipeline.jpg`) (if `--show-pipeline`)
   - All 8 processing steps in a single figure

## Console Output

The script prints detailed information about each processing step:

```
============================================================
ESP32-CAM Thresholding Object Detection (PC Version)
============================================================
Processing image: test_image.jpg
Image size: 640x480
Step 1: RGB to Grayscale conversion ✓
Step 2: Noise reduction (Gaussian blur) ✓
Step 3: Otsu threshold calculated: 145.00 ✓
Step 4: Binary thresholding ✓
Step 5: Small noise removal (morphological opening) ✓
Step 6: Connected components labeling: 3 components found ✓
Step 7: Blob properties calculated ✓
Step 8: Object detection: 1 objects found ✓
  - Object: 1002 pixels, Center: (320, 240), BBox: (200, 150) - (250, 190)

============================================================
✓ Successfully detected 1 object(s)
  Object 1:
    - Area: 1002 pixels
    - Center: (320, 240)
    - BBox: (200, 150, 250, 190)
============================================================
```

## Programmatic Usage

You can also use the functions directly in Python:

```python
from python_thresholding import threshold_and_detect_object, visualize_processing_pipeline

# Process image
result_image, detected_objects, processing_steps = threshold_and_detect_object(
    "input.jpg",
    object_size=1000,
    show_steps=True
)

# Visualize pipeline
visualize_processing_pipeline(processing_steps, "pipeline.jpg")

# Access detected objects
for obj in detected_objects:
    print(f"Area: {obj['area']} pixels")
    print(f"Center: {obj['centroid']}")
    print(f"BBox: {obj['bbox']}")
```

## Comparison with ESP32 Code

| Feature | ESP32-CAM | Python PC |
|---------|-----------|-----------|
| Image Size | 160×120 (QQVGA) | Any size |
| Noise Reduction | In-place 5-point filter | Gaussian blur |
| Threshold | Otsu (optimized) | Otsu (OpenCV) |
| Morphology | Custom 4-connected | OpenCV morphology |
| Components | Custom 2-pass | scipy.ndimage |
| Visualization | Serial output | Images + plots |
| Performance | ~200-300ms | ~50-100ms |

## Troubleshooting

### Import Errors

If you get import errors, install missing packages:
```bash
pip install opencv-python numpy scipy scikit-image matplotlib
```

### Image Not Found

The script will create a test image if the input image is not found:
```bash
python python_thresholding.py --create-test
```

### No Objects Detected

- Check if object size matches target (default: 1000 ± 50 pixels)
- Adjust `--target-size` and tolerance
- Try manual threshold with `--threshold`
- Check image contrast (object should be brighter than background)

### Memory Issues

For very large images, resize before processing:
```python
import cv2
image = cv2.imread("large_image.jpg")
image = cv2.resize(image, (640, 480))
cv2.imwrite("resized.jpg", image)
```

## Test Image Generation

The script can generate test images with known object sizes:

```python
from python_thresholding import create_test_image

# Create test image with ~1000 pixel object
create_test_image("test.jpg", object_size=1000)
```

## Notes

- The Python version uses more sophisticated algorithms (OpenCV, scikit-image)
- Results may differ slightly from ESP32 due to different implementations
- Use this for algorithm validation and parameter tuning
- Final implementation should match ESP32 constraints
