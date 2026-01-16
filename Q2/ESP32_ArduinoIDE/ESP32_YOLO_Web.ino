



/*
 * EE4065 - Final Project - Question 2
 * YOLO Digit Detection on ESP32-CAM
 * 
 * Features:
 * - Real-time digit detection (0, 3, 5, 8)
 * - Adaptive thresholding for MNIST-like preprocessing
 * - Bounding boxes with labels
 * - Modern web interface
 * 
 * Board: AI Thinker ESP32-CAM
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// TensorFlow Lite Micro (ESP32 version)
#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"        // <-- DOĞRUSU BU
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"


// ==================== WiFi ====================
const char* ssid = "mete";
const char* password = "12345678";

WebServer server(80);

// ==================== CAMERA PINS ====================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
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
#define FLASH_GPIO_NUM     4

// ==================== MODEL PARAMS ====================
#define IMG_SIZE          96
#define GRID_SIZE         12
#define NUM_ANCHORS       3
#define NUM_CLASSES       4
#define BOX_PARAMS        (5 + NUM_CLASSES)
#define CONFIDENCE_THRESHOLD  0.25f
#define MAX_DETECTIONS        10

const float ANCHORS[NUM_ANCHORS][2] = {{0.8f,0.8f},{1.2f,1.2f},{1.6f,1.6f}};
const int CLASS_DIGITS[NUM_CLASSES] = {0, 3, 5, 8};

// ==================== TFLITE ====================
constexpr int kTensorArenaSize = 320 * 1024;
uint8_t* tensor_arena = nullptr;

tflite::ErrorReporter* error_reporter = nullptr;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ==================== DETECTION ====================
struct Detection {
    int digit;
    int x1, y1, x2, y2;
    float confidence;
};
Detection detections[MAX_DETECTIONS];
int num_detections = 0;
unsigned long inference_time = 0;

// Display buffer
uint8_t* display_img = nullptr;
int display_w = 0, display_h = 0;
uint32_t frame_count = 0;

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Calculate IoU (Intersection over Union) for NMS
float calculateIoU(Detection& a, Detection& b) {
    int x1 = max(a.x1, b.x1);
    int y1 = max(a.y1, b.y1);
    int x2 = min(a.x2, b.x2);
    int y2 = min(a.y2, b.y2);
    
    if (x2 <= x1 || y2 <= y1) return 0;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    return intersection / (areaA + areaB - intersection);
}

// Non-Maximum Suppression
void applyNMS(float iouThreshold = 0.3f) {
    // Sort by confidence (bubble sort - simple for small arrays)
    for (int i = 0; i < num_detections - 1; i++) {
        for (int j = i + 1; j < num_detections; j++) {
            if (detections[j].confidence > detections[i].confidence) {
                Detection temp = detections[i];
                detections[i] = detections[j];
                detections[j] = temp;
            }
        }
    }
    
    // Mark suppressed detections
    bool suppressed[MAX_DETECTIONS] = {false};
    
    for (int i = 0; i < num_detections; i++) {
        if (suppressed[i]) continue;
        for (int j = i + 1; j < num_detections; j++) {
            if (suppressed[j]) continue;
            if (calculateIoU(detections[i], detections[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    
    // Compact array
    int newCount = 0;
    for (int i = 0; i < num_detections; i++) {
        if (!suppressed[i]) {
            detections[newCount++] = detections[i];
        }
    }
    num_detections = newCount;
}

// ==================== DRAW FUNCTIONS ====================
// Draw thick rectangle (4 pixels wide for visibility after downsampling)
void drawRect(uint8_t* img, int w, int h, int x1, int y1, int x2, int y2, uint8_t color) {
    x1 = max(2, min(x1, w-3));
    x2 = max(2, min(x2, w-3));
    y1 = max(2, min(y1, h-3));
    y2 = max(2, min(y2, h-3));
    
    // Draw 4-pixel thick lines
    for (int t = 0; t < 4; t++) {
        // Top line
        for (int x = x1; x <= x2; x++) {
            if (y1 + t < h) img[(y1 + t) * w + x] = color;
        }
        // Bottom line
        for (int x = x1; x <= x2; x++) {
            if (y2 - t >= 0) img[(y2 - t) * w + x] = color;
        }
        // Left line
        for (int y = y1; y <= y2; y++) {
            if (x1 + t < w) img[y * w + x1 + t] = color;
        }
        // Right line
        for (int y = y1; y <= y2; y++) {
            if (x2 - t >= 0) img[y * w + x2 - t] = color;
        }
    }
}

// Draw larger digit label (scaled 2x for visibility)
void drawDigit(uint8_t* img, int w, int h, int x, int y, int digit, uint8_t color) {
    const uint8_t digits[10][5] = {
        {0b111, 0b101, 0b101, 0b101, 0b111},  // 0
        {0b010, 0b110, 0b010, 0b010, 0b111},  // 1
        {0b111, 0b001, 0b111, 0b100, 0b111},  // 2
        {0b111, 0b001, 0b111, 0b001, 0b111},  // 3
        {0b101, 0b101, 0b111, 0b001, 0b001},  // 4
        {0b111, 0b100, 0b111, 0b001, 0b111},  // 5
        {0b111, 0b100, 0b111, 0b101, 0b111},  // 6
        {0b111, 0b001, 0b001, 0b001, 0b001},  // 7
        {0b111, 0b101, 0b111, 0b101, 0b111},  // 8
        {0b111, 0b101, 0b111, 0b001, 0b111}   // 9
    };
    
    if (digit < 0 || digit > 9) return;
    
    // Draw 2x scaled
    for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 3; dx++) {
            if (digits[digit][dy] & (1 << (2-dx))) {
                for (int sy = 0; sy < 2; sy++) {
                    for (int sx = 0; sx < 2; sx++) {
                        int px = x + dx*2 + sx;
                        int py = y + dy*2 + sy;
                        if (px >= 0 && px < w && py >= 0 && py < h) {
                            img[py * w + px] = color;
                        }
                    }
                }
            }
        }
    }
}

// ==================== DECODE DETECTIONS ====================
void decodeDetections() {
    num_detections = 0;
    
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    int8_t* output_data = output->data.int8;
    
    for (int gy = 0; gy < GRID_SIZE && num_detections < MAX_DETECTIONS; gy++) {
        for (int gx = 0; gx < GRID_SIZE && num_detections < MAX_DETECTIONS; gx++) {
            for (int a = 0; a < NUM_ANCHORS && num_detections < MAX_DETECTIONS; a++) {
                int offset = ((gy * GRID_SIZE + gx) * NUM_ANCHORS + a) * BOX_PARAMS;
                
                float tx = (output_data[offset + 0] - output_zero_point) * output_scale;
                float ty = (output_data[offset + 1] - output_zero_point) * output_scale;
                float tw = (output_data[offset + 2] - output_zero_point) * output_scale;
                float th = (output_data[offset + 3] - output_zero_point) * output_scale;
                float conf_raw = (output_data[offset + 4] - output_zero_point) * output_scale;
                
                float conf = sigmoid(conf_raw);
                if (conf < CONFIDENCE_THRESHOLD) continue;
                
                float max_class = -1000;
                int best_class = 0;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    float raw = (output_data[offset + 5 + c] - output_zero_point) * output_scale;
                    float score = sigmoid(raw);
                    if (score > max_class) { max_class = score; best_class = c; }
                }
                
                float final_conf = conf * max_class;
                if (final_conf < CONFIDENCE_THRESHOLD) continue;
                
                float bx = (sigmoid(tx) + gx) / GRID_SIZE;
                float by = (sigmoid(ty) + gy) / GRID_SIZE;
                float bw = (ANCHORS[a][0] * expf(tw)) / GRID_SIZE;
                float bh = (ANCHORS[a][1] * expf(th)) / GRID_SIZE;
                
                Detection det;
                det.digit = CLASS_DIGITS[best_class];
                det.x1 = (int)(fmax(0, bx - bw/2) * display_w);
                det.y1 = (int)(fmax(0, by - bh/2) * display_h);
                det.x2 = (int)(fmin(1, bx + bw/2) * display_w);
                det.y2 = (int)(fmin(1, by + bh/2) * display_h);
                det.confidence = final_conf;
                
                detections[num_detections++] = det;
            }
        }
    }
}

// ==================== CAPTURE AND DETECT ====================
void captureAndDetect() {
    // Warmup frames (reduced for speed)
    for (int i = 0; i < 2; i++) {
        camera_fb_t* warmup = esp_camera_fb_get();
        if (warmup) esp_camera_fb_return(warmup);
        delay(30);
    }
    
    // Get fresh frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { Serial.println("Capture failed!"); return; }
    
    display_w = fb->width;
    display_h = fb->height;
    frame_count++;
    
    // Allocate display buffer
    if (display_img) free(display_img);
    display_img = (uint8_t*)malloc(display_w * display_h);
    if (!display_img) {
        Serial.println("Display alloc failed!");
        esp_camera_fb_return(fb);
        return;
    }
    memcpy(display_img, fb->buf, display_w * display_h);
    
    // ========== FOMO-STYLE ADAPTIVE THRESHOLDING ==========
    // Calculate average brightness
    uint32_t sum = 0;
    for (int i = 0; i < IMG_SIZE * IMG_SIZE; i += 10) {
        sum += fb->buf[i];
    }
    uint8_t avg = sum / ((IMG_SIZE * IMG_SIZE) / 10);
    
    // Threshold: pixels darker than (avg - 50) are considered "ink"
    uint8_t threshold = (avg > 50) ? (avg - 50) : 0;
    
    Serial.printf("Frame %d: %dx%d, avg=%d, threshold=%d\n", 
                  frame_count, display_w, display_h, avg, threshold);
    
    // Direct copy with thresholding (no downsampling needed - camera is 96x96)
    // MNIST: white digits on black background
    // Camera: dark ink on bright paper
    for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
        uint8_t pixel = fb->buf[i];
        // If pixel is darker than threshold -> it's ink -> 127 (white)
        // Otherwise -> paper/shadow -> -128 (black)
        input->data.int8[i] = (pixel < threshold) ? 127 : -128;
    }
    
    esp_camera_fb_return(fb);
    
    // Run inference
    unsigned long start = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }
    inference_time = millis() - start;
    Serial.printf("Inference: %lu ms\n", inference_time);
    
    // Decode detections
    decodeDetections();
    
    // Apply Non-Maximum Suppression to remove overlapping boxes
    applyNMS(0.3f);
    
    // Draw bounding boxes on display image
    for (int i = 0; i < num_detections; i++) {
        Detection& det = detections[i];
        drawRect(display_img, display_w, display_h, det.x1, det.y1, det.x2, det.y2, 255);
        // Label above box (adjust for 2x scaled digit = 10px height)
        drawDigit(display_img, display_w, display_h, det.x1 + 4, det.y1 - 14, det.digit, 255);
        
        Serial.printf("Detected: %d (%.1f%%) at [%d,%d]-[%d,%d]\n", 
                      det.digit, det.confidence * 100,
                      det.x1, det.y1, det.x2, det.y2);
    }
    
    Serial.printf("Total detections: %d\n\n", num_detections);
}

// ==================== WEB HANDLERS ====================
// ==================== MODERN WEB ARAYÜZÜ ====================
const char* html = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32 Neural Vision</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #06b6d4; /* Cyan */
            --accent-glow: rgba(6, 182, 212, 0.3);
            --text-main: #e2e8f0;
            --text-dim: #94a3b8;
            --success: #10b981;
            --danger: #ef4444;
        }

        body {
            font-family: 'Courier New', Courier, monospace;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }

        .header {
            width: 100%;
            max-width: 800px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--accent);
            padding-bottom: 10px;
        }

        h1 { margin: 0; font-size: 1.5rem; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0 0 10px var(--accent-glow); }
        .status-badge { font-size: 0.8rem; background: var(--accent-glow); color: var(--accent); padding: 5px 10px; border-radius: 4px; border: 1px solid var(--accent); }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            width: 100%;
            max-width: 800px;
        }

        @media (min-width: 768px) {
            .main-grid { grid-template-columns: 3fr 2fr; }
        }

        .card {
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #334155;
            position: relative;
            overflow: hidden;
        }

        /* Viewfinder styling */
        .viewfinder-container {
            position: relative;
            aspect-ratio: 1;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid #334155;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .viewfinder-container.scanning { border-color: var(--accent); box-shadow: 0 0 15px var(--accent-glow); }
        
        #img { width: 100%; height: 100%; object-fit: contain; image-rendering: pixelated; }
        
        /* Scanline effect */
        .scan-line {
            position: absolute; width: 100%; height: 2px; background: var(--accent);
            opacity: 0.5; top: 0; left: 0; pointer-events: none; display: none;
            animation: scan 2s linear infinite;
        }
        @keyframes scan { 0% {top: 0;} 100% {top: 100%;} }

        /* Controls */
        .btn-group { display: flex; gap: 10px; margin-top: 15px; }
        
        button {
            flex: 1; padding: 12px; border: none; border-radius: 6px;
            font-family: inherit; font-weight: bold; cursor: pointer;
            transition: all 0.2s; text-transform: uppercase;
        }

        .btn-primary { background: var(--accent); color: #000; }
        .btn-primary:hover { background: #22d3ee; box-shadow: 0 0 10px var(--accent); }
        
        .btn-auto { background: #334155; color: var(--text-main); border: 1px solid var(--accent); }
        .btn-auto.active { background: var(--success); color: #000; border-color: var(--success); }

        /* Stats & Data */
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid #334155; padding-bottom: 5px; }
        .stat-label { color: var(--text-dim); font-size: 0.8rem; }
        .stat-value { color: var(--accent); font-weight: bold; }

        .log-container {
            height: 200px; overflow-y: auto; 
            background: rgba(0,0,0,0.2); border-radius: 6px; padding: 10px;
            font-size: 0.9rem; margin-top: 10px;
        }

        .log-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px; border-bottom: 1px solid #334155; animation: fadeIn 0.3s ease;
        }
        .digit-badge {
            background: var(--accent); color: #000; width: 24px; height: 24px;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-weight: bold;
        }
        
        .no-data { text-align: center; color: var(--text-dim); padding: 20px; font-style: italic; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>

    <div class="header">
        <h1>YOLO-V1 <span style="color:var(--accent); font-size:0.8em;">// TINY</span></h1>
        <div class="status-badge" id="sysStatus">SYSTEM READY</div>
    </div>

    <div class="main-grid">
        <div class="card">
            <div class="viewfinder-container" id="viewfinder">
                <div class="scan-line" id="scanLine"></div>
                <img id="img" src="/img" alt="Sensor Stream">
            </div>
            
            <div class="btn-group">
                <button class="btn-primary" id="btnDetect" onclick="singleShot()">Single Scan</button>
                <button class="btn-auto" id="btnAuto" onclick="toggleAuto()">Auto: OFF</button>
            </div>
        </div>

        <div class="card">
            <div style="margin-bottom: 15px;">
                <span style="color:var(--accent); font-weight:bold;">> TELEMETRY</span>
            </div>
            
            <div class="stat-row">
                <span class="stat-label">INFERENCE TIME</span>
                <span class="stat-value" id="timeVal">-- ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">DETECTIONS</span>
                <span class="stat-value" id="countVal">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">FRAME ID</span>
                <span class="stat-value" id="frameVal">0</span>
            </div>

            <div style="margin-top: 20px; margin-bottom: 10px;">
                <span style="color:var(--accent); font-weight:bold;">> DETECTION LOG</span>
            </div>
            <div class="log-container" id="logPanel">
                <div class="no-data">Waiting for input stream...</div>
            </div>
        </div>
    </div>

<script>
    let isAuto = false;
    let isProcessing = false;
    let refreshCount = 0;

    const img = document.getElementById('img');
    const scanLine = document.getElementById('scanLine');
    const logPanel = document.getElementById('logPanel');
    const sysStatus = document.getElementById('sysStatus');
    const viewfinder = document.getElementById('viewfinder');

    // Live View Loop (for when not detecting)
    setInterval(() => {
        if (!isProcessing && !isAuto) {
            refreshCount++;
            img.src = "/img?t=" + refreshCount;
        }
    }, 1500); // 1.5s refresh for idle view

    async function singleShot() {
        if (isProcessing) return;
        await runDetection();
    }

    function toggleAuto() {
        isAuto = !isAuto;
        const btn = document.getElementById('btnAuto');
        const btnDet = document.getElementById('btnDetect');
        
        if (isAuto) {
            btn.innerHTML = "Auto: ON";
            btn.classList.add('active');
            btnDet.disabled = true;
            btnDet.style.opacity = 0.5;
            runAutoLoop();
        } else {
            btn.innerHTML = "Auto: OFF";
            btn.classList.remove('active');
            btnDet.disabled = false;
            btnDet.style.opacity = 1;
        }
    }

    async function runAutoLoop() {
        if (!isAuto) return;
        await runDetection();
        if (isAuto) setTimeout(runAutoLoop, 100); // Small delay between scans
    }

    async function runDetection() {
        isProcessing = true;
        sysStatus.innerText = "PROCESSING...";
        viewfinder.classList.add('scanning');
        scanLine.style.display = 'block';

        try {
            const response = await fetch('/detect');
            const data = await response.json();
            
            // Update Image with Bounding Boxes
            refreshCount++;
            img.src = "/result?t=" + refreshCount;
            
            // Update Stats
            document.getElementById('timeVal').innerText = data.time + " ms";
            document.getElementById('countVal').innerText = data.count;
            document.getElementById('frameVal').innerText = data.frame;

            // Update Log
            updateLog(data.detections);

            sysStatus.innerText = "IDLE";
            sysStatus.style.color = "var(--accent)";

        } catch (error) {
            console.error(error);
            sysStatus.innerText = "ERROR";
            sysStatus.style.color = "var(--danger)";
        } finally {
            isProcessing = false;
            viewfinder.classList.remove('scanning');
            scanLine.style.display = 'none';
        }
    }

    function updateLog(detections) {
        if (detections.length === 0) {
            logPanel.innerHTML = '<div class="no-data">No objects detected</div>';
            return;
        }

        let html = '';
        detections.forEach(d => {
            const conf = (d.conf * 100).toFixed(1);
            html += `
            <div class="log-item">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div class="digit-badge">${d.digit}</div>
                    <span>Conf: ${conf}%</span>
                </div>
                <span style="font-size:0.8em; color:var(--text-dim);">[${d.x1},${d.y1}]</span>
            </div>`;
        });
        logPanel.innerHTML = html;
    }
</script>
</body>
</html>
)rawliteral";

void handleRoot() {
    server.send(200, "text/html", html);
}

void handleDetect() {
    captureAndDetect();
    
    String json = "{\"count\":" + String(num_detections);
    json += ",\"time\":" + String(inference_time);
    json += ",\"frame\":" + String(frame_count);
    json += ",\"detections\":[";
    
    for (int i = 0; i < num_detections; i++) {
        if (i > 0) json += ",";
        json += "{\"digit\":" + String(detections[i].digit);
        json += ",\"x1\":" + String(detections[i].x1);
        json += ",\"y1\":" + String(detections[i].y1);
        json += ",\"x2\":" + String(detections[i].x2);
        json += ",\"y2\":" + String(detections[i].y2);
        json += ",\"conf\":" + String(detections[i].confidence, 3) + "}";
    }
    json += "]}";
    
    server.send(200, "application/json", json);
}

void handleImage() {
    // Get live camera frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "No frame");
        return;
    }
    
    // Check if camera returned valid data
    if (fb->width == 0 || fb->height == 0 || fb->len == 0) {
        esp_camera_fb_return(fb);
        server.send(500, "text/plain", "Invalid frame");
        return;
    }
    
    // Camera is 96x96 - use simple BMP format like FOMO
    const int w = 96, h = 96;
    const int hdrSize = 54 + 256 * 4;
    const int imgSize = w * h;
    const int fileSize = hdrSize + imgSize;
    
    uint8_t* bmp = (uint8_t*)malloc(fileSize);
    if (!bmp) {
        esp_camera_fb_return(fb);
        server.send(500, "text/plain", "malloc fail");
        return;
    }
    
    memset(bmp, 0, fileSize);
    bmp[0] = 'B'; bmp[1] = 'M';
    *(uint32_t*)(bmp + 2) = fileSize;
    *(uint32_t*)(bmp + 10) = hdrSize;
    *(uint32_t*)(bmp + 14) = 40;
    *(int32_t*)(bmp + 18) = w;
    *(int32_t*)(bmp + 22) = h;
    *(uint16_t*)(bmp + 26) = 1;
    *(uint16_t*)(bmp + 28) = 8;
    *(uint32_t*)(bmp + 34) = imgSize;
    *(uint32_t*)(bmp + 46) = 256;
    
    // Grayscale palette
    for (int i = 0; i < 256; i++) {
        bmp[54 + i * 4 + 0] = i;
        bmp[54 + i * 4 + 1] = i;
        bmp[54 + i * 4 + 2] = i;
    }
    
    // Copy pixels (bottom-up for BMP) - 96 is divisible by 4, no padding needed
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bmp[hdrSize + (h - 1 - y) * w + x] = fb->buf[y * w + x];
        }
    }
    
    esp_camera_fb_return(fb);
    server.send_P(200, "image/bmp", (const char*)bmp, fileSize);
    free(bmp);
}

// Serve result image with bounding boxes (display_img buffer)
void handleResult() {
    if (!display_img || display_w == 0 || display_h == 0) {
        server.send(404, "text/plain", "No result yet - run detect first");
        return;
    }
    
    // Display is 96x96 - use simple BMP format
    const int w = 96, h = 96;
    const int hdrSize = 54 + 256 * 4;
    const int imgSize = w * h;
    const int fileSize = hdrSize + imgSize;
    
    uint8_t* bmp = (uint8_t*)malloc(fileSize);
    if (!bmp) {
        server.send(500, "text/plain", "malloc fail");
        return;
    }
    
    memset(bmp, 0, fileSize);
    bmp[0] = 'B'; bmp[1] = 'M';
    *(uint32_t*)(bmp + 2) = fileSize;
    *(uint32_t*)(bmp + 10) = hdrSize;
    *(uint32_t*)(bmp + 14) = 40;
    *(int32_t*)(bmp + 18) = w;
    *(int32_t*)(bmp + 22) = h;
    *(uint16_t*)(bmp + 26) = 1;
    *(uint16_t*)(bmp + 28) = 8;
    *(uint32_t*)(bmp + 34) = imgSize;
    *(uint32_t*)(bmp + 46) = 256;
    
    // Grayscale palette
    for (int i = 0; i < 256; i++) {
        bmp[54 + i * 4 + 0] = i;
        bmp[54 + i * 4 + 1] = i;
        bmp[54 + i * 4 + 2] = i;
    }
    
    // Copy display_img (with bounding boxes) - bottom-up for BMP
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bmp[hdrSize + (h - 1 - y) * w + x] = display_img[y * w + x];
        }
    }
    
    server.send_P(200, "image/bmp", (const char*)bmp, fileSize);
    free(bmp);
}

// ==================== CAMERA/TFLITE INIT ====================
bool initCamera() {
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
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_96X96;  // Direct model input size
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }
    
    sensor_t* s = esp_camera_sensor_get();
    s->set_brightness(s, 1);
    s->set_contrast(s, 2);
    
    return true;
}

bool initTFLite() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    
    model = tflite::GetModel(model_data);

    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed!");
        return false;
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.printf("TFLite ready! Arena: %d bytes\n", interpreter->arena_used_bytes());
    Serial.printf("Input: [%d,%d,%d,%d] type=%d\n",
        input->dims->data[0], input->dims->data[1],
        input->dims->data[2], input->dims->data[3], input->type);
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== EE4065 Q2: YOLO Digit Detection ===\n");
    
    pinMode(FLASH_GPIO_NUM, OUTPUT);
    digitalWrite(FLASH_GPIO_NUM, HIGH);
    
    // PSRAM allocation
    if (psramFound()) {
        tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
        Serial.printf("PSRAM: %d KB allocated\n", kTensorArenaSize / 1024);
    } else {
        Serial.println("No PSRAM!"); 
        while(1) delay(1000);
    }
    
    if (!initCamera()) { 
        Serial.println("Camera failed!"); 
        while(1) delay(1000); 
    }
    Serial.println("Camera OK!");
    
    if (!initTFLite()) { 
        Serial.println("TFLite failed!"); 
        while(1) delay(1000); 
    }
    Serial.println("TFLite OK!");
    
    // WiFi
    WiFi.begin(ssid, password);
    Serial.print("WiFi");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts++ < 30) {
        delay(500);
        Serial.print(".");
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nConnected: " + WiFi.localIP().toString());
    } else {
        WiFi.softAP("ESP32-YOLO", "12345678");
        Serial.println("\nAP Mode: " + WiFi.softAPIP().toString());
    }
    
    server.on("/", handleRoot);
    server.on("/detect", handleDetect);
    server.on("/img", handleImage);
    server.on("/result", handleResult);
    server.begin();
    
    // Initial capture
    captureAndDetect();
    
    Serial.println("\nReady! Open: http://" + 
        (WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : WiFi.softAPIP().toString()));
}

void loop() {
    server.handleClient();
    delay(1);
}
