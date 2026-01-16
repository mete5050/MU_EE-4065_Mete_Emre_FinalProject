/**
 * Question 1: Object Detection using Thresholding on ESP32-CAM
 * 
 * Detects a bright object (1000 pixels ±50) in an image where:
 * - Object pixels are brighter than background pixels
 * - Uses optimized Otsu thresholding algorithm
 * - Memory-optimized for ESP32-CAM constraints
 * 
 * Algorithm Pipeline:
 * 1. RGB565 to Grayscale conversion
 * 2. Noise reduction (in-place filtering)
 * 3. Otsu threshold calculation
 * 4. Binary thresholding
 * 5. Small noise removal
 * 6. Connected components labeling
 * 7. Blob analysis and object detection
 */

#include "esp_camera.h"
#include <Arduino.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// Target object size (pixels)
#define TARGET_OBJECT_SIZE 1000
#define SIZE_TOLERANCE 50

// Image dimensions (QQVGA - 160x120 for memory optimization)
#define IMAGE_WIDTH 160
#define IMAGE_HEIGHT 120

// Connected components labeling limits
#define MAX_LABELS 100
#define BACKGROUND 0

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Blob structure to store detected object properties
typedef struct {
    int area;                    // Area in pixels
    int min_x, min_y;           // Bounding box minimum coordinates
    int max_x, max_y;           // Bounding box maximum coordinates
    int centroid_x, centroid_y; // Centroid coordinates
} blob_t;

// ============================================================================
// CAMERA PIN DEFINITIONS (ESP32-CAM AI-Thinker Module)
// ============================================================================

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM     -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ============================================================================
// IMAGE PROCESSING FUNCTIONS
// ============================================================================

/**
 * Apply noise reduction using simple averaging filter (in-place)
 * Uses only a single row buffer to minimize memory usage
 * 
 * @param gray_image Input/output grayscale image
 * @param width Image width
 * @param height Image height
 */
void apply_noise_reduction(uint8_t *gray_image, int width, int height) {
    static uint8_t temp_line[IMAGE_WIDTH];  // Single row buffer
    
    // Apply 5-point cross filter (up, left, center, right, down)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            // Weighted average: center pixel has weight 2
            int sum = gray_image[(y-1) * width + x] +
                      gray_image[y * width + (x-1)] +
                      gray_image[y * width + x] * 2 +
                      gray_image[y * width + (x+1)] +
                      gray_image[(y+1) * width + x];
            temp_line[x] = (uint8_t)(sum / 6);
        }
        // Write back processed row (in-place operation)
        for (int x = 1; x < width - 1; x++) {
            gray_image[y * width + x] = temp_line[x];
        }
    }
}

/**
 * Calculate optimal threshold using Otsu's method
 * Finds threshold that maximizes inter-class variance
 * 
 * @param gray_image Input grayscale image
 * @param width Image width
 * @param height Image height
 * @return Optimal threshold value (0-255)
 */
int calculate_otsu_threshold(uint8_t *gray_image, int width, int height) {
    int histogram[256] = {0};
    int total_pixels = width * height;
    
    // Build histogram
    for (int i = 0; i < total_pixels; i++) {
        histogram[gray_image[i]]++;
    }
    
    // Otsu algorithm: maximize inter-class variance
    float sum = 0;           // Total weighted sum
    float sumB = 0;          // Sum for background class
    int q1 = 0;              // Background class pixel count
    float var_max = 0;       // Maximum variance
    int threshold = 128;     // Default threshold
    
    // Calculate total weighted sum
    for (int i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }
    
    // Find optimal threshold
    for (int t = 0; t < 256; t++) {
        q1 += histogram[t];
        if (q1 == 0) continue;
        
        int q2 = total_pixels - q1;
        if (q2 == 0) break;
        
        sumB += t * histogram[t];
        float m1 = sumB / q1;                    // Background mean
        float m2 = (sum - sumB) / q2;            // Foreground mean
        float var_between = (float)q1 * (float)q2 * (m1 - m2) * (m1 - m2);
        
        if (var_between > var_max) {
            var_max = var_between;
            threshold = t;
        }
    }
    
    // Adjust threshold for bright objects (slight boost if too low)
    if (threshold < 100) {
        threshold += 10;
    }
    
    return threshold;
}

/**
 * Convert RGB565 to grayscale
 * Uses standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
 * 
 * @param rgb565_image Input RGB565 image
 * @param gray_image Output grayscale image
 * @param width Image width
 * @param height Image height
 */
void rgb565_to_gray(uint16_t *rgb565_image, uint8_t *gray_image, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        uint16_t pixel = rgb565_image[i];
        // Extract RGB components from RGB565 format: RRRRR GGGGGG BBBBB
        uint8_t r = ((pixel >> 11) & 0x1F) << 3;
        uint8_t g = ((pixel >> 5) & 0x3F) << 2;
        uint8_t b = (pixel & 0x1F) << 3;
        // Calculate luminance
        gray_image[i] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

/**
 * Apply binary thresholding
 * Pixels above threshold become white (255), others become black (0)
 * 
 * @param gray_image Input grayscale image
 * @param binary_image Output binary image
 * @param width Image width
 * @param height Image height
 * @param threshold Threshold value
 */
void apply_threshold(uint8_t *gray_image, uint8_t *binary_image, int width, int height, int threshold) {
    for (int i = 0; i < width * height; i++) {
        binary_image[i] = (gray_image[i] > threshold) ? 255 : 0;
    }
}

/**
 * Remove isolated noise pixels (in-place)
 * Removes white pixels with no 4-connected neighbors
 * 
 * @param binary_image Input/output binary image
 * @param width Image width
 * @param height Image height
 */
void remove_small_noise(uint8_t *binary_image, int width, int height) {
    // Check 4-connected neighbors for each white pixel
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (binary_image[idx] == 255) {
                // Count 4-connected neighbors
                int neighbors = 0;
                if (binary_image[(y-1) * width + x] == 255) neighbors++;
                if (binary_image[y * width + (x-1)] == 255) neighbors++;
                if (binary_image[y * width + (x+1)] == 255) neighbors++;
                if (binary_image[(y+1) * width + x] == 255) neighbors++;
                
                // Remove isolated pixels (no neighbors)
                if (neighbors == 0) {
                    binary_image[idx] = 0;
                }
            }
        }
    }
    
    // Clear image borders
    for (int y = 0; y < height; y++) {
        binary_image[y * width + 0] = 0;
        binary_image[y * width + (width-1)] = 0;
    }
    for (int x = 0; x < width; x++) {
        binary_image[0 * width + x] = 0;
        binary_image[(height-1) * width + x] = 0;
    }
}

// ============================================================================
// CONNECTED COMPONENTS LABELING
// ============================================================================

/**
 * Label connected components using 4-connectivity
 * Uses two-pass algorithm with union-find for equivalence resolution
 * 
 * @param binary_image Input binary image
 * @param labels Output label image
 * @param width Image width
 * @param height Image height
 * @return Number of unique labels found
 */
int label_components(uint8_t *binary_image, uint16_t *labels, int width, int height) {
    int current_label = 1;
    static int equiv[MAX_LABELS];  // Equivalence table
    
    // Initialize equivalence table
    for (int i = 0; i < MAX_LABELS; i++) {
        equiv[i] = i;
    }
    
    // First pass: assign labels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            if (binary_image[idx] == 255) {
                int left_label = (x > 0) ? labels[idx - 1] : 0;
                int top_label = (y > 0) ? labels[(y - 1) * width + x] : 0;
                
                if (left_label == 0 && top_label == 0) {
                    // New component
                    if (current_label < MAX_LABELS) {
                        labels[idx] = current_label++;
                    } else {
                        labels[idx] = 0;  // Label limit exceeded
                    }
                } else if (left_label != 0 && top_label == 0) {
                    labels[idx] = left_label;
                } else if (left_label == 0 && top_label != 0) {
                    labels[idx] = top_label;
                } else {
                    // Merge labels (use minimum)
                    int min_label = (left_label < top_label) ? left_label : top_label;
                    int max_label = (left_label > top_label) ? left_label : top_label;
                    labels[idx] = min_label;
                    if (max_label < MAX_LABELS) {
                        equiv[max_label] = min_label;
                    }
                }
            } else {
                labels[idx] = BACKGROUND;
            }
        }
    }
    
    // Resolve equivalence table (path compression)
    for (int i = 1; i < current_label && i < MAX_LABELS; i++) {
        int root = i;
        while (equiv[root] != root && root < MAX_LABELS) {
            root = equiv[root];
        }
        equiv[i] = root;
    }
    
    // Second pass: merge equivalent labels
    for (int i = 0; i < width * height; i++) {
        if (labels[i] != BACKGROUND && labels[i] < MAX_LABELS) {
            labels[i] = equiv[labels[i]];
        }
    }
    
    return current_label - 1;
}

/**
 * Calculate blob properties (area, centroid, bounding box)
 * 
 * @param labels Input label image
 * @param width Image width
 * @param height Image height
 * @param blobs Output blob array
 * @param max_blobs Maximum number of blobs to process
 * @return Number of blobs found
 */
int calculate_blob_properties(uint16_t *labels, int width, int height, blob_t *blobs, int max_blobs) {
    static uint16_t area[MAX_LABELS] = {0};
    static uint16_t sum_x[MAX_LABELS] = {0};
    static uint16_t sum_y[MAX_LABELS] = {0};
    static uint8_t min_x_arr[MAX_LABELS];
    static uint8_t min_y_arr[MAX_LABELS];
    static uint8_t max_x_arr[MAX_LABELS];
    static uint8_t max_y_arr[MAX_LABELS];
    
    // Reset arrays
    for (int i = 0; i < MAX_LABELS; i++) {
        area[i] = 0;
        sum_x[i] = 0;
        sum_y[i] = 0;
        min_x_arr[i] = (uint8_t)width;
        min_y_arr[i] = (uint8_t)height;
        max_x_arr[i] = 0;
        max_y_arr[i] = 0;
    }
    
    // Accumulate blob statistics
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int label = labels[idx];
            
            if (label != BACKGROUND && label < MAX_LABELS) {
                area[label]++;
                sum_x[label] += x;
                sum_y[label] += y;
                
                if (x < min_x_arr[label]) min_x_arr[label] = (uint8_t)x;
                if (x > max_x_arr[label]) max_x_arr[label] = (uint8_t)x;
                if (y < min_y_arr[label]) min_y_arr[label] = (uint8_t)y;
                if (y > max_y_arr[label]) max_y_arr[label] = (uint8_t)y;
            }
        }
    }
    
    // Fill blob structures
    int blob_count = 0;
    for (int i = 1; i < MAX_LABELS && blob_count < max_blobs; i++) {
        if (area[i] > 0) {
            blobs[blob_count].area = area[i];
            blobs[blob_count].centroid_x = sum_x[i] / area[i];
            blobs[blob_count].centroid_y = sum_y[i] / area[i];
            blobs[blob_count].min_x = min_x_arr[i];
            blobs[blob_count].min_y = min_y_arr[i];
            blobs[blob_count].max_x = max_x_arr[i];
            blobs[blob_count].max_y = max_y_arr[i];
            blob_count++;
        }
    }
    
    return blob_count;
}

// ============================================================================
// MAIN DETECTION FUNCTION
// ============================================================================

/**
 * Main object detection function
 * Performs complete thresholding pipeline and detects target object
 * 
 * @param detected_blob Output: detected blob if found
 * @return 0 if object found, -1 otherwise
 */
int detect_object_by_size(blob_t *detected_blob) {
    sensor_t *s = esp_camera_sensor_get();
    if (s == NULL) {
        Serial.println("ERROR: Sensor not found");
        return -1;
    }
    
    // Switch to RGB565 format for processing
    int format_result = s->set_pixformat(s, PIXFORMAT_RGB565);
    int size_result = s->set_framesize(s, FRAMESIZE_QQVGA);
    
    if (format_result != 0 || size_result != 0) {
        Serial.println("ERROR: Failed to set camera format");
        return -1;
    }
    
    delay(100);  // Allow format change to settle
    
    // Capture frame
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb == NULL) {
        Serial.println("ERROR: Camera frame buffer NULL");
        return -1;
    }
    
    if (fb->format != PIXFORMAT_RGB565) {
        Serial.println("ERROR: RGB565 format not received");
        esp_camera_fb_return(fb);
        return -1;
    }
    
    int width = fb->width;
    int height = fb->height;
    
    // Static buffers (memory optimized)
    static uint8_t gray_image[IMAGE_WIDTH * IMAGE_HEIGHT];
    static uint8_t binary_image[IMAGE_WIDTH * IMAGE_HEIGHT];
    static uint16_t labels[IMAGE_WIDTH * IMAGE_HEIGHT];
    
    if (width > IMAGE_WIDTH || height > IMAGE_HEIGHT) {
        Serial.println("ERROR: Image size too large");
        esp_camera_fb_return(fb);
        return -1;
    }
    
    // Processing pipeline
    // Step 1: Convert RGB565 to grayscale
    rgb565_to_gray((uint16_t *)fb->buf, gray_image, width, height);
    
    // Step 2: Apply noise reduction (in-place)
    apply_noise_reduction(gray_image, width, height);
    
    // Step 3: Calculate Otsu threshold
    int threshold = calculate_otsu_threshold(gray_image, width, height);
    Serial.print("Calculated threshold: ");
    Serial.println(threshold);
    
    // Step 4: Apply binary thresholding
    apply_threshold(gray_image, binary_image, width, height, threshold);
    
    // Step 5: Remove small noise (in-place)
    remove_small_noise(binary_image, width, height);
    
    // Step 6: Label connected components
    int num_labels = label_components(binary_image, labels, width, height);
    Serial.print("Components detected: ");
    Serial.println(num_labels);
    
    // Step 7: Calculate blob properties
    static blob_t blobs[MAX_LABELS];
    int blob_count = calculate_blob_properties(labels, width, height, blobs, MAX_LABELS);
    
    // Step 8: Find object matching target size
    int found = 0;
    for (int i = 0; i < blob_count; i++) {
        int size_diff = abs(blobs[i].area - TARGET_OBJECT_SIZE);
        if (size_diff <= SIZE_TOLERANCE) {
            *detected_blob = blobs[i];
            found = 1;
            Serial.print("Object detected! Size: ");
            Serial.print(blobs[i].area);
            Serial.print(" pixels, Center: (");
            Serial.print(blobs[i].centroid_x);
            Serial.print(", ");
            Serial.print(blobs[i].centroid_y);
            Serial.println(")");
            break;
        }
    }
    
    esp_camera_fb_return(fb);
    
    return found ? 0 : -1;
}

// ============================================================================
// ARDUINO SETUP AND LOOP
// ============================================================================

void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(false);
    Serial.println();
    Serial.println("ESP32-CAM Thresholding Object Detection");
    Serial.println("========================================");
    
    // Camera configuration
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QQVGA;  // 160x120
    config.fb_count = 1;
    
    // Initialize camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera initialization error: 0x%x\n", err);
        return;
    }
    
    Serial.println("Camera initialized successfully");
    Serial.println("Processing pipeline:");
    Serial.println("  1. RGB565 -> Grayscale conversion");
    Serial.println("  2. Noise reduction (in-place)");
    Serial.println("  3. Otsu threshold calculation");
    Serial.println("  4. Binary thresholding");
    Serial.println("  5. Small noise removal");
    Serial.println("  6. Connected components labeling");
    Serial.println("  7. Blob analysis");
    Serial.println("========================================");
    delay(1000);
}

void loop() {
    // Run detection every 1 second
    static unsigned long last_threshold_time = 0;
    unsigned long current_time = millis();
    
    if (current_time - last_threshold_time >= 1000) {
        last_threshold_time = current_time;
        
        blob_t detected_blob;
        int result = detect_object_by_size(&detected_blob);
        
        if (result == 0) {
            Serial.println("=== OBJECT DETECTED ===");
            Serial.print("Size: ");
            Serial.print(detected_blob.area);
            Serial.println(" pixels");
            Serial.print("Center: (");
            Serial.print(detected_blob.centroid_x);
            Serial.print(", ");
            Serial.print(detected_blob.centroid_y);
            Serial.println(")");
            Serial.print("Bounding Box: (");
            Serial.print(detected_blob.min_x);
            Serial.print(", ");
            Serial.print(detected_blob.min_y);
            Serial.print(") - (");
            Serial.print(detected_blob.max_x);
            Serial.print(", ");
            Serial.print(detected_blob.max_y);
            Serial.println(")");
            Serial.println("========================");
        } else {
            Serial.println("Target object (1000±50 pixels) not detected");
        }
    }
    
    delay(10);
}
