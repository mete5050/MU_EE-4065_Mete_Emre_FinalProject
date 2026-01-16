"""
Question 1a: PC-based Thresholding for Object Detection

Assignment Requirements:
- One bright object in the image (background pixels are darker)
- Object to be detected has 1000 pixels
- Thresholding result should extract the object based on its size

This Python code runs on PC to perform thresholding operation:
1. Capture image from camera
2. Apply thresholding to separate bright object from dark background
3. Extract object based on size (1000 pixels)

Requirements:
    pip install opencv-python numpy scipy scikit-image
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure
import argparse
import time
from collections import deque


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_OBJECT_SIZE = 1000  # Target object size in pixels (as per assignment requirements)
SIZE_TOLERANCE = 50  # Size tolerance in pixels (reasonable default for 1000px target)
PROCESS_INTERVAL = 0.1  # Process every 100ms for better performance
CONSOLE_PRINT_INTERVAL = 1.0  # Print to console every 1 second


# ============================================================================
# THRESHOLDING FUNCTIONS
# ============================================================================

def calculate_otsu_threshold(gray_image):
    """Calculate optimal threshold using Otsu's method"""
    threshold_value, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_value


def apply_threshold(gray_image, threshold_value):
    """Apply binary thresholding"""
    _, binary = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


def remove_noise(binary_image):
    """Remove small noise using morphological opening"""
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def label_connected_components(binary_image):
    """Label connected components using 4-connectivity"""
    labeled_image, num_features = ndimage.label(binary_image, structure=np.ones((3, 3)))
    return labeled_image, num_features


def calculate_blob_properties(labeled_image):
    """Calculate blob properties"""
    regions = measure.regionprops(labeled_image)
    return regions


def detect_object_by_size(regions, target_size=TARGET_OBJECT_SIZE, tolerance=SIZE_TOLERANCE, show_all=False):
    """
    Find objects matching target size
    
    Args:
        regions: List of region properties
        target_size: Target object size in pixels
        tolerance: Size tolerance in pixels
        show_all: If True, return all blobs with their distances from target
    
    Returns:
        detected_objects: List of detected objects
        all_blobs: List of all blobs (if show_all=True)
    """
    detected_objects = []
    all_blobs = []
    
    for region in regions:
        area = region.area
        size_diff = abs(area - target_size)
        min_row, min_col, max_row, max_col = region.bbox
        centroid = region.centroid
        
        blob_info = {
            'area': area,
            'centroid': (int(centroid[1]), int(centroid[0])),
            'bbox': (min_col, min_row, max_col, max_row),
            'size_diff': size_diff,
            'is_target': size_diff <= tolerance
        }
        
        all_blobs.append(blob_info)
        
        if size_diff <= tolerance:
            detected_objects.append(blob_info)
    
    # Always return tuple for consistency
    return detected_objects, all_blobs


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_frame(frame, threshold_value=None):
    """
    Process a single frame and return results
    
    Args:
        frame: Input frame (BGR)
        threshold_value: Manual threshold (None for Otsu)
    
    Returns:
        detected_objects, threshold_used, num_components, binary_image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate threshold
    if threshold_value is None:
        threshold_used = calculate_otsu_threshold(gray)
    else:
        threshold_used = threshold_value
    
    # Apply binary thresholding
    binary = apply_threshold(gray, threshold_used)
    
    # Remove noise
    cleaned = remove_noise(binary)
    
    # Label connected components
    labeled_image, num_components = label_connected_components(cleaned)
    
    # Calculate blob properties
    regions = calculate_blob_properties(labeled_image)
    
    # Detect objects by size
    detected_objects, _ = detect_object_by_size(regions, TARGET_OBJECT_SIZE, SIZE_TOLERANCE, show_all=False)
    
    return detected_objects, threshold_used, num_components, cleaned, regions


# ============================================================================
# CAMERA DETECTION
# ============================================================================

def run_camera_detection(camera_id=0):
    """Run real-time detection from webcam with improved performance"""
    print("=" * 70)
    print("ESP32-CAM Thresholding Object Detection - Real-time Camera Mode")
    print("=" * 70)
    print(f"Target object size: {TARGET_OBJECT_SIZE} ± {SIZE_TOLERANCE} pixels")
    print("\nControls:")
    print("  [q] Quit")
    print("  [s] Save current frame")
    print("  [r] Reset threshold (Otsu)")
    print("  [+/-] Adjust threshold ±5")
    print("  [w/x] Adjust target size ±50")
    print("  [a/d] Adjust tolerance ±10")
    print("  [b] Show/hide ALL blobs")
    print("  [h] Show/hide help")
    print("=" * 70)
    
    # Open camera
    print(f"\nOpening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    # Try alternative backends
    if not cap.isOpened():
        print("Trying DirectShow backend...")
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Trying MSMF backend...")
        cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
    
    if not cap.isOpened():
        print(f"\n❌ ERROR: Could not open camera {camera_id}")
        print("\nTroubleshooting:")
        print("  1. Check if camera is connected")
        print("  2. Try different camera ID:")
        print("     python python_thresholding.py --camera 1")
        print("  3. Close other programs using the camera")
        print("  4. On Windows, try running as administrator")
        
        # Scan for available cameras
        print("\nScanning for available cameras...")
        available = []
        for test_id in range(5):
            test_cap = cv2.VideoCapture(test_id)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    available.append(test_id)
                test_cap.release()
        
        if available:
            print(f"✓ Found {len(available)} available camera(s): {available}")
            print(f"  Try: python python_thresholding.py --camera {available[0]}")
        else:
            print("✗ No cameras found")
        return
    
    print(f"✓ Camera {camera_id} opened successfully")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    print("\nStarting detection... (Show your bright object to camera)")
    print("=" * 70)
    
    # State variables
    frame_count = 0
    save_counter = 0
    last_process_time = time.time()
    last_print_time = time.time()
    current_threshold = None
    current_target_size = TARGET_OBJECT_SIZE  # Start with default
    current_tolerance = SIZE_TOLERANCE  # Start with default
    show_help = False
    
    # FPS calculation
    fps_buffer = deque(maxlen=30)
    fps_start_time = time.time()
    
    # Cached results
    last_detected_objects = []
    last_threshold = 0
    last_num_components = 0
    last_binary_image = None
    last_all_blobs = []
    show_all_blobs = True  # Show all blobs by default
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("❌ Error: Failed to read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Calculate FPS
            fps_buffer.append(current_time)
            if len(fps_buffer) > 1:
                fps = len(fps_buffer) / (fps_buffer[-1] - fps_buffer[0])
            else:
                fps = 0
            
            # Process frame at intervals (performance optimization)
            process_frame_now = (current_time - last_process_time >= PROCESS_INTERVAL)
            
            if process_frame_now:
                last_process_time = current_time
                
                # Process frame
                detected_objects, threshold_used, num_components, binary_image, regions = process_frame(
                    frame, threshold_value=current_threshold
                )
                
                # Get all blobs for visualization
                detected_objs, all_blobs = detect_object_by_size(regions, current_target_size, current_tolerance, show_all=True)
                
                # Update cached results
                last_detected_objects = detected_objs
                last_threshold = threshold_used
                last_num_components = num_components
                last_binary_image = binary_image
                last_all_blobs = all_blobs
                
                # Update threshold if using manual
                if current_threshold is not None:
                    current_threshold = threshold_used
            
            # Draw results on frame
            result_frame = frame.copy()
            
            # Draw all blobs (always show all for better visibility)
            blobs_to_draw = last_all_blobs if len(last_all_blobs) > 0 else last_detected_objects
            
            for obj in blobs_to_draw:
                x_min, y_min, x_max, y_max = obj['bbox']
                
                # Choose color based on whether it matches target
                if obj.get('is_target', False):
                    box_color = (0, 255, 0)  # Green for target
                    text_color = (0, 255, 0)
                else:
                    box_color = (255, 165, 0)  # Orange for other blobs
                    text_color = (255, 255, 0)
                
                # Bounding box
                thickness = 3 if obj.get('is_target', False) else 2
                cv2.rectangle(result_frame, (x_min, y_min), (x_max, y_max), box_color, thickness)
                
                # Centroid
                cx, cy = obj['centroid']
                cv2.circle(result_frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.circle(result_frame, (cx, cy), 8, (0, 0, 255), 2)
                
                # Label with area and difference
                label = f"{obj['area']}px (diff: {obj['size_diff']})"
                cv2.putText(result_frame, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(result_frame, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # Add info panel with better formatting
            info_lines = [
                f"Frame: {frame_count}",
                f"FPS: {fps:.1f}",
                f"Threshold: {last_threshold:.1f}",
                f"Components: {last_num_components}",
                f"Objects: {len(last_detected_objects)}",
                f"Target: {current_target_size}±{current_tolerance}px"
            ]
            
            # Show blob information
            if len(last_all_blobs) > 0:
                info_lines.append(f"All blobs: {len(last_all_blobs)}")
                # Show closest blob info
                closest = min(last_all_blobs, key=lambda x: x['size_diff'])
                info_lines.append(f"Closest: {closest['area']}px")
                info_lines.append(f"  (diff: {closest['size_diff']})")
            
            # Draw info panel background
            panel_height = len(info_lines) * 25 + 10
            cv2.rectangle(result_frame, (5, 5), (250, panel_height), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (5, 5), (250, panel_height), (255, 255, 255), 1)
            
            y_pos = 25
            for line in info_lines:
                cv2.putText(result_frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
            
            # Status text with better visibility
            if last_detected_objects:
                status = f"DETECTED! ({len(last_detected_objects)} object(s))"
                color = (0, 255, 0)
            else:
                status = "No object detected"
                color = (0, 0, 255)
            
            # Status background
            text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result_frame, (5, height - 35), (text_size[0] + 15, height - 5), (0, 0, 0), -1)
            cv2.putText(result_frame, status, (10, height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Help overlay
            if show_help:
                help_lines = [
                    "KEYBOARD COMMANDS:",
                    "[q] Quit",
                    "[s] Save frame",
                    "[r] Reset threshold",
                    "[+/-] Threshold +/-5",
                    "[w/x] Target size +/-50",
                    "[a/d] Tolerance +/-10",
                    "[b] Show all blobs",
                    "[h] Hide help"
                ]
                help_y = height - len(help_lines) * 20 - 10
                cv2.rectangle(result_frame, (width - 220, help_y - 5), 
                             (width - 5, height - 5), (0, 0, 0), -1)
                cv2.rectangle(result_frame, (width - 220, help_y - 5), 
                             (width - 5, height - 5), (255, 255, 0), 2)
                for i, line in enumerate(help_lines):
                    cv2.putText(result_frame, line, (width - 215, help_y + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Display frames
            cv2.imshow('Camera - Object Detection', result_frame)
            if last_binary_image is not None:
                cv2.imshow('Binary Threshold', last_binary_image)
            
            # Print to console at intervals
            if current_time - last_print_time >= CONSOLE_PRINT_INTERVAL:
                last_print_time = current_time
                print(f"\nFrame {frame_count} (FPS: {fps:.1f}):")
                print(f"  Threshold: {last_threshold:.1f}")
                print(f"  Components: {last_num_components}")
                
                if last_detected_objects:
                    print("  === OBJECT DETECTED ===")
                    for i, obj in enumerate(last_detected_objects, 1):
                        print(f"  Object {i}:")
                        print(f"    Size: {obj['area']} pixels (diff: {obj['size_diff']})")
                        print(f"    Center: ({obj['centroid'][0]}, {obj['centroid'][1]})")
                        print(f"    BBox: ({obj['bbox'][0]}, {obj['bbox'][1]}) - ({obj['bbox'][2]}, {obj['bbox'][3]})")
                    print("  ========================")
                else:
                    print(f"  {current_target_size}±{current_tolerance} pixel object not detected")
                    # Show all blobs for debugging
                    if len(last_all_blobs) > 0:
                        print(f"  All blobs found ({len(last_all_blobs)}):")
                        # Sort by size difference
                        sorted_blobs = sorted(last_all_blobs, key=lambda x: x['size_diff'])
                        for i, blob in enumerate(sorted_blobs[:5], 1):  # Show top 5 closest
                            print(f"    Blob {i}: {blob['area']}px (diff: {blob['size_diff']})")
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                save_counter += 1
                filename = f"capture_{save_counter:04d}.jpg"
                cv2.imwrite(filename, result_frame)
                if last_binary_image is not None:
                    cv2.imwrite(filename.replace('.jpg', '_binary.jpg'), last_binary_image)
                print(f"\n✓ Frame saved: {filename}")
            elif key == ord('r'):
                current_threshold = None
                print("Threshold reset to Otsu (automatic)")
            elif key == ord('+') or key == ord('='):
                if current_threshold is None:
                    current_threshold = last_threshold
                current_threshold = min(255, current_threshold + 5)
                print(f"Threshold increased to: {current_threshold}")
            elif key == ord('-') or key == ord('_'):
                if current_threshold is None:
                    current_threshold = last_threshold
                current_threshold = max(0, current_threshold - 5)
                print(f"Threshold decreased to: {current_threshold}")
            elif key == ord('w'):
                current_target_size = max(50, current_target_size + 50)
                print(f"Target size increased to: {current_target_size}px")
            elif key == ord('x'):
                current_target_size = max(50, current_target_size - 50)
                print(f"Target size decreased to: {current_target_size}px")
            elif key == ord('a'):
                current_tolerance = min(200, current_tolerance + 10)
                print(f"Tolerance increased to: ±{current_tolerance}px")
            elif key == ord('d'):
                current_tolerance = max(10, current_tolerance - 10)
                print(f"Tolerance decreased to: ±{current_tolerance}px")
            elif key == ord('h'):
                show_help = not show_help
                print(f"Help {'shown' if show_help else 'hidden'}")
            elif key == ord('b'):
                show_all_blobs = not show_all_blobs
                print(f"Show all blobs: {'ON' if show_all_blobs else 'OFF'}")
                if show_all_blobs and len(last_all_blobs) > 0:
                    print(f"  Found {len(last_all_blobs)} blobs:")
                    for i, blob in enumerate(last_all_blobs[:10], 1):  # Show first 10
                        print(f"    Blob {i}: {blob['area']}px (diff: {blob['size_diff']})")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Exiting...")


# ============================================================================
# IMAGE FILE PROCESSING
# ============================================================================

def process_image_file(image_path, output_path=None, threshold_value=None):
    """Process image file with detailed output"""
    print("=" * 70)
    print("ESP32-CAM Thresholding Object Detection - Image File Mode")
    print("=" * 70)
    print(f"Target object size: {TARGET_OBJECT_SIZE} ± {SIZE_TOLERANCE} pixels")
    print("=" * 70)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return None
    
    print(f"\nProcessing: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print()
    
    # Process with step-by-step output
    print("Step 1: RGB to Grayscale conversion ✓")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("Step 2: Calculating threshold...")
    if threshold_value is None:
        threshold = calculate_otsu_threshold(gray)
        print(f"Step 2: Otsu threshold calculated: {threshold:.2f} ✓")
    else:
        threshold = threshold_value
        print(f"Step 2: Using manual threshold: {threshold:.2f} ✓")
    
    print("Step 3: Applying binary thresholding ✓")
    binary = apply_threshold(gray, threshold)
    
    print("Step 4: Removing noise (morphological opening) ✓")
    cleaned = remove_noise(binary)
    
    print("Step 5: Labeling connected components...")
    labeled_image, num_components = label_connected_components(cleaned)
    print(f"Step 5: Connected components labeling: {num_components} components found ✓")
    
    print("Step 6: Calculating blob properties ✓")
    regions = calculate_blob_properties(labeled_image)
    
    print("Step 7: Detecting objects by size...")
    detected_objects = detect_object_by_size(regions, TARGET_OBJECT_SIZE, SIZE_TOLERANCE)
    print(f"Step 7: Object detection: {len(detected_objects)} objects found ✓")
    print()
    
    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Threshold: {threshold:.2f}")
    print(f"Components: {num_components}")
    
    if detected_objects:
        print("\n=== OBJECT DETECTED ===")
        for i, obj in enumerate(detected_objects, 1):
            print(f"Object {i}:")
            print(f"  Size: {obj['area']} pixels (diff: {obj['size_diff']})")
            print(f"  Center: ({obj['centroid'][0]}, {obj['centroid'][1]})")
            print(f"  BBox: ({obj['bbox'][0]}, {obj['bbox'][1]}) - ({obj['bbox'][2]}, {obj['bbox'][3]})")
        print("========================")
    else:
        print(f"\n{TARGET_OBJECT_SIZE}±{SIZE_TOLERANCE} pixel object not detected")
        # Show all blobs for debugging
        if len(regions) > 0:
            print("\nAll detected blobs (for debugging):")
            for i, region in enumerate(regions[:10], 1):  # Show first 10
                print(f"  Blob {i}: {region.area} pixels")
    print("=" * 70)
    
    # Draw results
    result_image = image.copy()
    for obj in detected_objects:
        x_min, y_min, x_max, y_max = obj['bbox']
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cx, cy = obj['centroid']
        cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)
        label = f"Area: {obj['area']}px"
        cv2.putText(result_image, label, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, result_image)
        cv2.imwrite(output_path.replace('.jpg', '_binary.jpg'), cleaned)
        print(f"\n✓ Result saved: {output_path}")
        print(f"✓ Binary image saved: {output_path.replace('.jpg', '_binary.jpg')}")
    else:
        cv2.imshow('Detection Result', result_image)
        cv2.imshow('Binary Image', cleaned)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detected_objects
    

def create_test_image(output_path="test_image.jpg"):
    """Create synthetic test image with bright object"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    import math
    side = int(math.sqrt(TARGET_OBJECT_SIZE))
    x = (640 - side) // 2
    y = (480 - side) // 2
    cv2.rectangle(image, (x, y), (x + side, y + side), (200, 200, 200), -1)
    noise = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    cv2.imwrite(output_path, image)
    print(f"✓ Test image created: {output_path}")
    print(f"  Object size: ~{side * side} pixels")
    print(f"  Object position: ({x}, {y}) - ({x + side}, {y + side})")
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Real-time Object Detection using Thresholding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python python_thresholding.py --camera          # Real-time from webcam
  python python_thresholding.py -i image.jpg      # Process image file
  python python_thresholding.py --create-test     # Create test image
        """
    )
    parser.add_argument('--camera', '-c', type=int, default=None,
                       help='Use webcam (camera ID, default: 0)')
    parser.add_argument('--image', '-i', type=str, default=None,
                       help='Process image file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output image path')
    parser.add_argument('--create-test', action='store_true',
                       help='Create synthetic test image')
    
    args = parser.parse_args()
    
    # Camera mode (default if no arguments)
    if args.camera is not None or (args.image is None and not args.create_test):
        camera_id = args.camera if args.camera is not None else 0
        run_camera_detection(camera_id)
        return 0
    
    # Create test image
    if args.create_test:
        test_path = create_test_image("test_image.jpg")
        process_image_file(test_path, "test_result.jpg")
        return 0
    
    # Process image file
    if args.image:
        process_image_file(args.image, args.output)
        return 0
    
    # Default: open camera
    run_camera_detection(0)
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user (Ctrl+C)")
        exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
