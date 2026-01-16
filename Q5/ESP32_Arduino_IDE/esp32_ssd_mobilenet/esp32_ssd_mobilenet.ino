/*
 * EE4065 Final Project - Question 5 (Bonus)
 * Handwritten Digit Detection (FOMO / SSD) on ESP32-CAM
 * * Features:
 * - Object Detection utilizing custom trained model.h
 * - Supports FOMO (Grid-based) or SSD outputs
 * - Modern "Neural Dashboard" Web Interface
 * * Note: Ensure your 'model.h' variable matches 'model_data' extern.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>

// TensorFlow Lite Micro
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ====================== USER CONFIG ======================
// 1. Include your trained model header
#include "model.h" 

// 2. DEFINE YOUR MODEL VARIABLE NAME HERE
// Check inside model.h (e.g., const unsigned char g_model[]...)
// If it is named 'g_model', change the line below to: extern const unsigned char g_model[];
extern const unsigned char model_data[]; 
#define MODEL_PTR model_data

// 3. Network Config
const char* WIFI_SSID = "mete";
const char* WIFI_PASS = "12345678";

// 4. Model Parameters (Update these based on your training!)
#define INPUT_W         96    // Width expected by model (e.g. 96, 160, 320)
#define INPUT_H         96    // Height expected by model
#define NUM_CLASSES     11    // 0-9 + Background (or just 10 depending on training)
#define CONF_THRESH     0.5f  // Detection threshold

// ====================== PIN DEFINITIONS ======================
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

// ====================== GLOBALS ======================
httpd_handle_t webServer = NULL;

// TFLite Globals
const int kTensorArenaSize = 256 * 1024; // Adjust if model is large
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Display/Processing Buffers
uint8_t* display_buf = nullptr; // For web streaming
size_t display_len = 0;

// Detection Structure
struct ObjectDet {
    int id;
    int x;
    int y;
    int w;
    int h;
    float conf;
};
#define MAX_DETS 20
ObjectDet results[MAX_DETS];
int result_count = 0;
unsigned long inference_time = 0;

// ====================== TFLITE SETUP ======================
bool setupModel() {
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::AllOpsResolver resolver;

    const tflite::Model* model = tflite::GetModel(MODEL_PTR);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return false;
    }

    if (psramFound()) {
        tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    } else {
        tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed!");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.printf("Model Loaded. Input: %d x %d x %d\n", 
        input->dims->data[1], input->dims->data[2], input->dims->data[3]);
    return true;
}

// ====================== CAMERA SETUP ======================
bool setupCamera() {
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
    config.pixel_format = PIXFORMAT_RGB565; // Use RGB for models usually
    config.frame_size = FRAMESIZE_QVGA;     // 320x240
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    if (esp_camera_init(&config) != ESP_OK) return false;
    
    sensor_t *s = esp_camera_sensor_get();
    s->set_vflip(s, 1); // Flip if necessary
    s->set_hmirror(s, 1);

    return true;
}

// ====================== PREPROCESSING ======================
// Crop center and resize to Model Input Size
void processImage(camera_fb_t* fb) {
    int src_w = fb->width;
    int src_h = fb->height;
    uint16_t* src_buf = (uint16_t*)fb->buf;

    // Center crop calculation
    int min_side = (src_w < src_h) ? src_w : src_h;
    int start_x = (src_w - min_side) / 2;
    int start_y = (src_h - min_side) / 2;

    // Resize ratio
    // We treat input as int8 (-128 to 127) or uint8 depending on model
    // Assuming int8 quantized for ESP32 efficiency
    
    for (int y = 0; y < INPUT_H; y++) {
        for (int x = 0; x < INPUT_W; x++) {
            // Nearest neighbor interpolation
            int sx = start_x + (x * min_side / INPUT_W);
            int sy = start_y + (y * min_side / INPUT_H);
            
            uint16_t pixel = src_buf[sy * src_w + sx];
            
            // Extract RGB
            uint8_t r = (pixel >> 11) & 0x1F; r = (r * 255) / 31;
            uint8_t g = (pixel >> 5) & 0x3F;  g = (g * 255) / 63;
            uint8_t b = pixel & 0x1F;         b = (b * 255) / 31;

            // Normalize and Assign to Tensor
            // Standard quantization: (val / 255.0) / scale + zero_point
            // Simplified for signed int8: val - 128
            
            if (input->type == kTfLiteInt8) {
                input->data.int8[(y * INPUT_W + x) * 3 + 0] = (int8_t)(r - 128);
                input->data.int8[(y * INPUT_W + x) * 3 + 1] = (int8_t)(g - 128);
                input->data.int8[(y * INPUT_W + x) * 3 + 2] = (int8_t)(b - 128);
            } else { // Float
                 input->data.f[(y * INPUT_W + x) * 3 + 0] = r / 255.0f;
                 input->data.f[(y * INPUT_W + x) * 3 + 1] = g / 255.0f;
                 input->data.f[(y * INPUT_W + x) * 3 + 2] = b / 255.0f;
            }
        }
    }
}

// ====================== POST-PROCESSING (FOMO) ======================
// FOMO outputs a grid (e.g. 12x12xClasses). We find peaks in this grid.
void postProcessFOMO() {
    result_count = 0;
    
    // Output dimensions: [1, GridY, GridX, Classes]
    int grid_h = output->dims->data[1];
    int grid_w = output->dims->data[2];
    int classes = output->dims->data[3];
    
    // Calculate stride (how many pixels one grid cell represents)
    int stride_x = INPUT_W / grid_w;
    int stride_y = INPUT_H / grid_h;

    int8_t* out_data = output->data.int8; // Assuming int8 output
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            float max_conf = 0;
            int max_class = -1;

            // Find best class for this grid cell (skip class 0 if it's background)
            // Note: Check your training. Usually 0 is background in FOMO.
            for (int c = 1; c < classes; c++) { 
                int val = out_data[(y * grid_w * classes) + (x * classes) + c];
                float conf = (val - zero_point) * scale; // Dequantize
                
                if (conf > max_conf) {
                    max_conf = conf;
                    max_class = c;
                }
            }

            if (max_conf >= CONF_THRESH && result_count < MAX_DETS) {
                // FOMO Detection Found (Centroid)
                results[result_count].id = max_class;
                results[result_count].conf = max_conf;
                // Map grid back to pixel coordinates
                results[result_count].x = x * stride_x + (stride_x / 2);
                results[result_count].y = y * stride_y + (stride_y / 2);
                results[result_count].w = stride_x; // Visual size
                results[result_count].h = stride_y;
                result_count++;
            }
        }
    }
}

void runInference() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return;

    // Prepare buffer for web (convert RGB565 to JPEG later or send as is)
    // For simplicity, we process the RGB565 directly
    processImage(fb);
    
    // Keep frame for web streaming (optional: convert here if needed)
    // We just return it now to free memory for inference
    esp_camera_fb_return(fb);

    unsigned long start = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed");
        return;
    }
    inference_time = millis() - start;

    // Parse Results
    postProcessFOMO();
    
    Serial.printf("Inf: %lums, Objects: %d\n", inference_time, result_count);
}

// ====================== WEB SERVER ======================
// Modern Cyberpunk/Amber Dashboard for "Bonus" feel
const char* HTML_PAGE = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32 Object Vision</title>
    <style>
        :root {
            --bg: #121212;
            --card: #1e1e1e;
            --accent: #f59e0b; /* Amber for Bonus */
            --accent-glow: rgba(245, 158, 11, 0.2);
            --text: #e0e0e0;
            --grid: #333;
        }
        body { font-family: 'Segoe UI', monospace; background: var(--bg); color: var(--text); margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        h1 { border-bottom: 2px solid var(--accent); padding-bottom: 10px; margin-bottom: 20px; letter-spacing: 2px; }
        
        .main-container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; max-width: 1200px; width: 100%; }
        
        .card { background: var(--card); padding: 15px; border-radius: 12px; border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        
        /* Camera View with Overlay */
        .cam-wrapper { position: relative; width: 320px; height: 320px; background: #000; border-radius: 8px; overflow: hidden; border: 2px solid var(--accent); }
        #camImg { width: 100%; height: 100%; object-fit: contain; }
        #overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
        
        .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; }
        button { background: #333; color: var(--text); border: 1px solid var(--accent); padding: 12px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: 0.3s; }
        button:hover { background: var(--accent); color: #000; box-shadow: 0 0 10px var(--accent-glow); }
        
        .stats-panel { min-width: 300px; }
        .stat-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #333; }
        .stat-val { color: var(--accent); font-weight: bold; }
        
        .log-list { height: 200px; overflow-y: auto; font-size: 0.9em; margin-top: 10px; }
        .log-item { padding: 5px; border-left: 2px solid var(--accent); margin-bottom: 5px; background: rgba(255,255,255,0.05); }

        /* Loading Scanline */
        .scan { position: absolute; width: 100%; height: 2px; background: var(--accent); top: 0; animation: scanAnim 2s infinite linear; opacity: 0.5; }
        @keyframes scanAnim { 0% {top:0;} 100% {top:100%;} }
    </style>
</head>
<body>
    <h1>Q5: BONUS VISION <span style="font-size:0.5em; color:#777;">// FOMO & SSD</span></h1>
    
    <div class="main-container">
        <div class="card">
            <div class="cam-wrapper">
                <div class="scan"></div>
                <img id="camImg" src="">
                <canvas id="overlay" width="96" height="96"></canvas> </div>
            <div class="controls">
                <button onclick="toggleAuto()" id="btnAuto">AUTO DETECT: OFF</button>
                <button onclick="runOnce()">SINGLE SCAN</button>
            </div>
        </div>

        <div class="card stats-panel">
            <div style="color:var(--accent); margin-bottom:10px;">SYSTEM TELEMETRY</div>
            <div class="stat-row"><span>Status</span><span id="status" class="stat-val">IDLE</span></div>
            <div class="stat-row"><span>Inference</span><span id="infTime" class="stat-val">0 ms</span></div>
            <div class="stat-row"><span>Objects</span><span id="objCount" class="stat-val">0</span></div>
            
            <div style="color:var(--accent); margin-top:20px; margin-bottom:5px;">DETECTION LOG</div>
            <div class="log-list" id="log"></div>
        </div>
    </div>

<script>
    let isAuto = false;
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    
    // Scale canvas logic to display size
    // Note: We draw on 96x96 canvas but CSS stretches it to 320x320
    // This keeps coordinate logic simple matching the model input.

    function log(msg) {
        const d = document.getElementById('log');
        d.innerHTML = `<div class="log-item">${msg}</div>` + d.innerHTML;
    }

    async function runOnce() {
        document.getElementById('status').innerText = "INFERENCING...";
        try {
            // 1. Trigger Inference
            const resp = await fetch('/detect');
            const data = await resp.json();
            
            // 2. Update Image (Refresh from buffer)
            document.getElementById('camImg').src = "/capture?t=" + Date.now();
            
            // 3. Update Stats
            document.getElementById('infTime').innerText = data.time + " ms";
            document.getElementById('objCount').innerText = data.count;
            document.getElementById('status').innerText = "READY";
            
            // 4. Draw Results
            drawObjects(data.objects);
            
        } catch(e) {
            console.error(e);
            document.getElementById('status').innerText = "ERROR";
        }
    }

    function drawObjects(objects) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        objects.forEach(obj => {
            // Draw Point/Box
            ctx.strokeStyle = "#f59e0b";
            ctx.fillStyle = "rgba(245, 158, 11, 0.3)";
            ctx.lineWidth = 2;
            
            // FOMO returns centroids. We draw a small box or circle.
            // If SSD, 'w' and 'h' would be actual bounding box sizes.
            const size = obj.w > 0 ? obj.w : 8; 
            
            ctx.beginPath();
            ctx.rect(obj.x - size/2, obj.y - size/2, size, size);
            ctx.stroke();
            ctx.fill();
            
            log(`Class ${obj.id} @ [${obj.x},${obj.y}] (${(obj.conf*100).toFixed(0)}%)`);
        });
    }

    function toggleAuto() {
        isAuto = !isAuto;
        document.getElementById('btnAuto').innerText = isAuto ? "AUTO DETECT: ON" : "AUTO DETECT: OFF";
        document.getElementById('btnAuto').style.background = isAuto ? "#f59e0b" : "#333";
        document.getElementById('btnAuto').style.color = isAuto ? "#000" : "#e0e0e0";
        if(isAuto) loop();
    }

    async function loop() {
        if(!isAuto) return;
        await runOnce();
        if(isAuto) setTimeout(loop, 200);
    }
    
    // Initial Load
    document.getElementById('camImg').src = "/capture";
</script>
</body>
</html>
)rawliteral";

// ====================== HANDLERS ======================
esp_err_t handleRoot(httpd_req_t *req) {
    httpd_resp_send(req, HTML_PAGE, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

esp_err_t handleCapture(httpd_req_t *req) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) return ESP_FAIL;
    
    // Send basic JPEG
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return ESP_OK;
}

esp_err_t handleDetect(httpd_req_t *req) {
    runInference();
    
    char json[1024];
    char objs[800] = "[";
    
    for(int i=0; i<result_count; i++) {
        char item[64];
        sprintf(item, "{\"id\":%d,\"x\":%d,\"y\":%d,\"w\":%d,\"h\":%d,\"conf\":%.2f}%s",
            results[i].id, results[i].x, results[i].y, results[i].w, results[i].h, results[i].conf,
            (i < result_count - 1) ? "," : "");
        strcat(objs, item);
    }
    strcat(objs, "]");
    
    sprintf(json, "{\"time\":%lu,\"count\":%d,\"objects\":%s}", inference_time, result_count, objs);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, strlen(json));
    return ESP_OK;
}

void startServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_start(&webServer, &config);
    
    httpd_uri_t uri_root = { "/", HTTP_GET, handleRoot, NULL };
    httpd_uri_t uri_cap = { "/capture", HTTP_GET, handleCapture, NULL };
    httpd_uri_t uri_det = { "/detect", HTTP_GET, handleDetect, NULL };
    
    httpd_register_uri_handler(webServer, &uri_root);
    httpd_register_uri_handler(webServer, &uri_cap);
    httpd_register_uri_handler(webServer, &uri_det);
}

void setup() {
    Serial.begin(115200);
    Serial.println("Starting Q5 Bonus Project...");

    setupCamera();
    setupModel();

    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println(WiFi.localIP());

    startServer();
}

void loop() {
    delay(100);
}