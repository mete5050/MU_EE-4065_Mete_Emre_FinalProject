

#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// TFLite Micro Libraries
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model Weights - All models in single header
#include "model_data.h"

// ====================== Network Settings ======================
const char* WIFI_SSID = "mete";
const char* WIFI_PASS = "12345678";
const char* AP_SSID = "meteeker";
const char* AP_PASS = "mete2002";

// ====================== Hardware Pins (AI-THINKER) ======================
#define CAM_PIN_PWDN     32
#define CAM_PIN_RESET    -1
#define CAM_PIN_XCLK      0
#define CAM_PIN_SIOD     26
#define CAM_PIN_SIOC     27
#define CAM_PIN_D7       35
#define CAM_PIN_D6       34
#define CAM_PIN_D5       39
#define CAM_PIN_D4       36
#define CAM_PIN_D3       21
#define CAM_PIN_D2       19
#define CAM_PIN_D1       18
#define CAM_PIN_D0        5
#define CAM_PIN_VSYNC    25
#define CAM_PIN_HREF     23
#define CAM_PIN_PCLK     22
#define LED_FLASH         4

// ====================== Model Configuration ======================
enum ModelType { 
    MDL_SQUEEZE = 0, 
    MDL_MOBILE = 1, 
    MDL_RESNET = 2, 
    MDL_EFFICIENT = 3,
    MDL_ENSEMBLE = 4 
};

#define IMG_INPUT_DIM 32
#define IMG_CHANNELS 3
#define DIGIT_CLASSES 10
#define ARENA_BYTES (92 * 1024)
#define PREPROCESS_DIM 160

// Model name strings
const char* MODEL_LABELS[] = {"SqueezeNet", "MobileNetV2", "ResNet-8", "EfficientNet", "Ensemble"};

// ====================== Global Variables ======================
httpd_handle_t webServer = NULL;
uint8_t* tensorMemory = nullptr;

const tflite::Model* activeModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* inputLayer = nullptr;
TfLiteTensor* outputLayer = nullptr;

int activeModelIdx = MDL_SQUEEZE;
float predictionProbs[5][DIGIT_CLASSES];
int predictedDigits[5];
float confidenceScores[5];
uint32_t inferenceMs[5];

uint8_t processedPreview[IMG_INPUT_DIM * IMG_INPUT_DIM];
uint8_t* grayBuffer = nullptr;
uint8_t* binaryBuffer = nullptr;
uint8_t* morphBuffer = nullptr;

// ====================== Model Data Accessors ======================
const unsigned char* getModelBytes(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_model;
        case MDL_MOBILE:    return mobilenetv2mini_model;
        case MDL_RESNET:    return resnet8_model;
        case MDL_EFFICIENT: return efficientnetmini_model;
        default: return squeezenetmini_model;
    }
}

unsigned int getModelSize(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_model_len;
        case MDL_MOBILE:    return mobilenetv2mini_model_len;
        case MDL_RESNET:    return resnet8_model_len;
        case MDL_EFFICIENT: return efficientnetmini_model_len;
        default: return squeezenetmini_model_len;
    }
}

float getInScale(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_input_scale;
        case MDL_MOBILE:    return mobilenetv2mini_input_scale;
        case MDL_RESNET:    return resnet8_input_scale;
        case MDL_EFFICIENT: return efficientnetmini_input_scale;
        default: return 0.003921569f;
    }
}

int getInZeroPoint(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_input_zero_point;
        case MDL_MOBILE:    return mobilenetv2mini_input_zero_point;
        case MDL_RESNET:    return resnet8_input_zero_point;
        case MDL_EFFICIENT: return efficientnetmini_input_zero_point;
        default: return 0;
    }
}

float getOutScale(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_output_scale;
        case MDL_MOBILE:    return mobilenetv2mini_output_scale;
        case MDL_RESNET:    return resnet8_output_scale;
        case MDL_EFFICIENT: return efficientnetmini_output_scale;
        default: return 0.00390625f;
    }
}

int getOutZeroPoint(int idx) {
    switch(idx) {
        case MDL_SQUEEZE:   return squeezenetmini_output_zero_point;
        case MDL_MOBILE:    return mobilenetv2mini_output_zero_point;
        case MDL_RESNET:    return resnet8_output_zero_point;
        case MDL_EFFICIENT: return efficientnetmini_output_zero_point;
        default: return -128;
    }
}

// ====================== TFLite Initialization ======================
static tflite::MicroMutableOpResolver<20> opResolver;

bool setupTFLite(int modelIdx) {
    Serial.printf("[TFLite] Loading %s model...\n", MODEL_LABELS[modelIdx]);
    
    activeModel = tflite::GetModel(getModelBytes(modelIdx));
    if (activeModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[TFLite] Schema version mismatch!");
        return false;
    }
    
    // Register required operations (once)
    static bool opsReady = false;
    if (!opsReady) {
        opResolver.AddConv2D();
        opResolver.AddDepthwiseConv2D();
        opResolver.AddMaxPool2D();
        opResolver.AddAveragePool2D();
        opResolver.AddReshape();
        opResolver.AddSoftmax();
        opResolver.AddRelu();
        opResolver.AddRelu6();
        opResolver.AddAdd();
        opResolver.AddMul();
        opResolver.AddMean();
        opResolver.AddPad();
        opResolver.AddConcatenation();
        opResolver.AddQuantize();
        opResolver.AddDequantize();
        opResolver.AddLogistic();
        opResolver.AddFullyConnected();
        opsReady = true;
    }
    
    static tflite::MicroErrorReporter errReporter;
    static tflite::MicroInterpreter staticInterp(
        activeModel, opResolver, tensorMemory, ARENA_BYTES, &errReporter);
    tflInterpreter = &staticInterp;
    
    if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TFLite] Tensor allocation failed!");
        return false;
    }
    
    inputLayer = tflInterpreter->input(0);
    outputLayer = tflInterpreter->output(0);
    
    Serial.printf("[TFLite] Input shape: [%d,%d,%d,%d]\n", 
        inputLayer->dims->data[0], inputLayer->dims->data[1],
        inputLayer->dims->data[2], inputLayer->dims->data[3]);
    Serial.printf("[TFLite] Arena usage: %d bytes\n", tflInterpreter->arena_used_bytes());
    
    activeModelIdx = modelIdx;
    return true;
}

// ====================== Camera Setup ======================
bool setupCamera() {
    camera_config_t camCfg;
    camCfg.ledc_channel = LEDC_CHANNEL_0;
    camCfg.ledc_timer = LEDC_TIMER_0;
    camCfg.pin_d0 = CAM_PIN_D0;
    camCfg.pin_d1 = CAM_PIN_D1;
    camCfg.pin_d2 = CAM_PIN_D2;
    camCfg.pin_d3 = CAM_PIN_D3;
    camCfg.pin_d4 = CAM_PIN_D4;
    camCfg.pin_d5 = CAM_PIN_D5;
    camCfg.pin_d6 = CAM_PIN_D6;
    camCfg.pin_d7 = CAM_PIN_D7;
    camCfg.pin_xclk = CAM_PIN_XCLK;
    camCfg.pin_pclk = CAM_PIN_PCLK;
    camCfg.pin_vsync = CAM_PIN_VSYNC;
    camCfg.pin_href = CAM_PIN_HREF;
    camCfg.pin_sscb_sda = CAM_PIN_SIOD;
    camCfg.pin_sscb_scl = CAM_PIN_SIOC;
    camCfg.pin_pwdn = CAM_PIN_PWDN;
    camCfg.pin_reset = CAM_PIN_RESET;
    camCfg.xclk_freq_hz = 20000000;
    camCfg.pixel_format = PIXFORMAT_RGB565;
    camCfg.frame_size = FRAMESIZE_QVGA;
    camCfg.jpeg_quality = 10;
    camCfg.fb_count = 2;
    camCfg.fb_location = CAMERA_FB_IN_PSRAM;
    camCfg.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    
    if (esp_camera_init(&camCfg) != ESP_OK) {
        Serial.println("[Camera] Initialization failed!");
        return false;
    }
    
    Serial.println("[Camera] Ready");
    return true;
}

// ====================== Image Preprocessing ======================
// Custom preprocessing: center crop, Otsu threshold, morphology, resize
void preprocessFrame(camera_fb_t* frame, int8_t* modelInput, int modelIdx) {
    uint16_t* rgb565 = (uint16_t*)frame->buf;
    int srcW = frame->width;
    int srcH = frame->height;
    
    float inScale = getInScale(modelIdx);
    int inZP = getInZeroPoint(modelIdx);
    
    // Step 1: Center crop and downsample to 160x160 grayscale
    int cropDim = min(srcW, srcH);
    int offsetX = (srcW - cropDim) / 2;
    int offsetY = (srcH - cropDim) / 2;
    float scaleRatio = (float)cropDim / PREPROCESS_DIM;
    
    uint8_t minPix = 255, maxPix = 0;
    
    for (int y = 0; y < PREPROCESS_DIM; y++) {
        for (int x = 0; x < PREPROCESS_DIM; x++) {
            int sx = offsetX + (int)(x * scaleRatio);
            int sy = offsetY + (int)(y * scaleRatio);
            
            uint16_t pix = rgb565[sy * srcW + sx];
            // Extract RGB565 and convert to grayscale
            uint8_t r = (pix >> 11) & 0x1F;
            uint8_t g = (pix >> 5) & 0x3F;
            uint8_t b = pix & 0x1F;
            uint8_t gray = (r * 77 + g * 150 + b * 29) >> 8;
            
            grayBuffer[y * PREPROCESS_DIM + x] = gray;
            if (gray < minPix) minPix = gray;
            if (gray > maxPix) maxPix = gray;
        }
    }
    
    // Step 2: Contrast stretching
    int range = maxPix - minPix;
    if (range < 10) range = 10;
    
    for (int i = 0; i < PREPROCESS_DIM * PREPROCESS_DIM; i++) {
        int stretched = ((grayBuffer[i] - minPix) * 255) / range;
        grayBuffer[i] = constrain(stretched, 0, 255);
    }
    
    // Step 3: FOMO-style Adaptive Thresholding
    // Calculate average brightness (like FOMO)
    uint32_t sumBrightness = 0;
    for (int i = 0; i < PREPROCESS_DIM * PREPROCESS_DIM; i++) {
        sumBrightness += grayBuffer[i];
    }
    uint8_t avgBrightness = sumBrightness / (PREPROCESS_DIM * PREPROCESS_DIM);
    
    // Adaptive threshold: avg - 30 (same as FOMO)
    // Pixels darker than threshold are "ink" (become white in MNIST format)
    int adaptiveThreshold = avgBrightness - 30;
    if (adaptiveThreshold < 10) adaptiveThreshold = 10;
    
    Serial.printf("[Preprocess] Avg brightness: %d, Threshold: %d\n", avgBrightness, adaptiveThreshold);
    
    // Step 4: Create binary image (white digit on black background)
    // Camera: dark ink on light paper
    // MNIST: white digit (255) on black background (0)
    for (int i = 0; i < PREPROCESS_DIM * PREPROCESS_DIM; i++) {
        // If pixel is darker than threshold -> it's ink -> make it white (255)
        // Otherwise -> it's paper -> make it black (0)
        binaryBuffer[i] = (grayBuffer[i] < adaptiveThreshold) ? 255 : 0;
    }
    
    // Step 5: Dilation (thicken digit strokes)
    memcpy(morphBuffer, binaryBuffer, PREPROCESS_DIM * PREPROCESS_DIM);
    for (int y = 1; y < PREPROCESS_DIM - 1; y++) {
        for (int x = 1; x < PREPROCESS_DIM - 1; x++) {
            int idx = y * PREPROCESS_DIM + x;
            if (binaryBuffer[idx] || binaryBuffer[idx-1] || binaryBuffer[idx+1] ||
                binaryBuffer[idx-PREPROCESS_DIM] || binaryBuffer[idx+PREPROCESS_DIM]) {
                morphBuffer[idx] = 255;
            }
        }
    }
    memcpy(binaryBuffer, morphBuffer, PREPROCESS_DIM * PREPROCESS_DIM);
    
    // Step 6: Find bounding box
    int minX = PREPROCESS_DIM, minY = PREPROCESS_DIM, maxX = -1, maxY = -1;
    int digitPixels = 0;
    
    for (int y = 0; y < PREPROCESS_DIM; y++) {
        for (int x = 0; x < PREPROCESS_DIM; x++) {
            if (binaryBuffer[y * PREPROCESS_DIM + x]) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                digitPixels++;
            }
        }
    }
    
    // Fallback if no digit found
    if (digitPixels < 20) {
        minX = 0; minY = 0; 
        maxX = PREPROCESS_DIM - 1; 
        maxY = PREPROCESS_DIM - 1;
    }
    
    // Add 15% padding
    int padX = (maxX - minX + 1) * 15 / 100;
    int padY = (maxY - minY + 1) * 15 / 100;
    minX = max(0, minX - padX);
    minY = max(0, minY - padY);
    maxX = min(PREPROCESS_DIM - 1, maxX + padX);
    maxY = min(PREPROCESS_DIM - 1, maxY + padY);
    
    int boxW = maxX - minX + 1;
    int boxH = maxY - minY + 1;
    
    // Step 7: Resize bounding box to 32x32 and quantize
    for (int y = 0; y < IMG_INPUT_DIM; y++) {
        for (int x = 0; x < IMG_INPUT_DIM; x++) {
            int sx = minX + (x * boxW) / IMG_INPUT_DIM;
            int sy = minY + (y * boxH) / IMG_INPUT_DIM;
            sx = constrain(sx, 0, PREPROCESS_DIM - 1);
            sy = constrain(sy, 0, PREPROCESS_DIM - 1);
            
            uint8_t pixVal = binaryBuffer[sy * PREPROCESS_DIM + sx];
            processedPreview[y * IMG_INPUT_DIM + x] = pixVal;
            
            // Quantize: normalize to [0,1] then apply scale/zero_point
            float normalized = pixVal / 255.0f;
            int8_t quantized = (int8_t)(normalized / inScale + inZP);
            
            // Replicate grayscale to RGB channels
            modelInput[(y * IMG_INPUT_DIM + x) * 3 + 0] = quantized;
            modelInput[(y * IMG_INPUT_DIM + x) * 3 + 1] = quantized;
            modelInput[(y * IMG_INPUT_DIM + x) * 3 + 2] = quantized;
        }
    }
}

// ====================== Inference ======================
int runPrediction(int modelIdx) {
    // Discard stale frame
    camera_fb_t* oldFrame = esp_camera_fb_get();
    if (oldFrame) esp_camera_fb_return(oldFrame);
    
    // Capture fresh frame
    camera_fb_t* frame = esp_camera_fb_get();
    if (!frame) {
        Serial.println("[Inference] Camera capture failed!");
        return -1;
    }
    
    // Switch model if needed
    if (activeModelIdx != modelIdx) {
        if (!setupTFLite(modelIdx)) {
            esp_camera_fb_return(frame);
            return -1;
        }
    }
    
    // Preprocess
    preprocessFrame(frame, inputLayer->data.int8, modelIdx);
    esp_camera_fb_return(frame);
    
    // Run inference
    uint32_t startTime = millis();
    if (tflInterpreter->Invoke() != kTfLiteOk) {
        Serial.println("[Inference] Failed!");
        return -1;
    }
    inferenceMs[modelIdx] = millis() - startTime;
    
    // Decode output
    float outScale = getOutScale(modelIdx);
    int outZP = getOutZeroPoint(modelIdx);
    
    int bestClass = 0;
    float bestScore = -1000;
    
    for (int c = 0; c < DIGIT_CLASSES; c++) {
        float prob = (outputLayer->data.int8[c] - outZP) * outScale;
        predictionProbs[modelIdx][c] = prob;
        if (prob > bestScore) {
            bestScore = prob;
            bestClass = c;
        }
    }
    
    predictedDigits[modelIdx] = bestClass;
    confidenceScores[modelIdx] = bestScore;
    
    Serial.printf("[%s] Predicted: %d (%.1f%%) in %lu ms\n",
        MODEL_LABELS[modelIdx], bestClass, bestScore * 100, inferenceMs[modelIdx]);
    
    return bestClass;
}

// Ensemble: run all models and average probabilities
int runEnsemble() {
    Serial.println("\n[Ensemble] Running all models...");
    
    for (int m = 0; m < 4; m++) {
        runPrediction(m);
    }
    
    // Average probabilities
    float avgProbs[DIGIT_CLASSES] = {0};
    for (int c = 0; c < DIGIT_CLASSES; c++) {
        for (int m = 0; m < 4; m++) {
            avgProbs[c] += predictionProbs[m][c];
        }
        avgProbs[c] /= 4.0f;
    }
    
    // Find best
    int bestClass = 0;
    float bestProb = avgProbs[0];
    for (int c = 1; c < DIGIT_CLASSES; c++) {
        if (avgProbs[c] > bestProb) {
            bestProb = avgProbs[c];
            bestClass = c;
        }
    }
    
    for (int c = 0; c < DIGIT_CLASSES; c++) {
        predictionProbs[MDL_ENSEMBLE][c] = avgProbs[c];
    }
    predictedDigits[MDL_ENSEMBLE] = bestClass;
    confidenceScores[MDL_ENSEMBLE] = bestProb;
    inferenceMs[MDL_ENSEMBLE] = inferenceMs[0] + inferenceMs[1] + inferenceMs[2] + inferenceMs[3];
    
    Serial.printf("[Ensemble] Final: %d (%.1f%%) total %lu ms\n",
        bestClass, bestProb * 100, inferenceMs[MDL_ENSEMBLE]);
    
    return bestClass;
}
// ==================== MODERN CNN ANALYTICS DASHBOARD ====================
const char* WEB_PAGE = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Model CNN Lab</title>
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --accent: #38bdf8; /* Sky Blue */
            --accent-glow: rgba(56, 189, 248, 0.3);
            --ensemble: #c084fc; /* Purple for Ensemble */
            --text: #e2e8f0;
            --dim: #94a3b8;
            --success: #4ade80;
        }

        body {
            font-family: 'Courier New', Courier, monospace;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }

        .header {
            width: 100%; max-width: 1000px;
            display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid var(--accent); margin-bottom: 20px; padding-bottom: 10px;
        }
        h1 { margin: 0; font-size: 1.4rem; letter-spacing: 1px; }
        .badge { background: var(--accent-glow); color: var(--accent); padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; border: 1px solid var(--accent); }

        /* GRID LAYOUT */
        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            width: 100%; max-width: 1000px;
        }
        
        .full-width { grid-column: span 2; }

        @media (max-width: 768px) {
            .grid-container { grid-template-columns: 1fr; }
            .full-width { grid-column: span 1; }
        }

        .card {
            background: var(--card); border-radius: 12px; padding: 15px;
            border: 1px solid #334155; box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }

        /* CAMERA SECTION */
        .view-row { display: flex; gap: 10px; margin-bottom: 15px; }
        .cam-box { 
            flex: 2; position: relative; border-radius: 8px; overflow: hidden; 
            border: 2px solid #334155; aspect-ratio: 1; background: #000;
        }
        .proc-box { 
            flex: 1; position: relative; border-radius: 8px; overflow: hidden; 
            border: 2px solid var(--accent); aspect-ratio: 1; background: #000;
        }
        img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .label { position: absolute; bottom: 0; left: 0; background: rgba(0,0,0,0.7); width: 100%; text-align: center; font-size: 0.7rem; padding: 2px 0; }

        /* CONTROLS */
        .model-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 10px; }
        button {
            padding: 12px; border: none; border-radius: 6px; cursor: pointer;
            font-family: inherit; font-weight: bold; font-size: 0.8rem;
            background: #334155; color: var(--dim); transition: 0.2s;
        }
        button:hover { background: #475569; color: var(--text); }
        button.active { background: var(--accent); color: #000; box-shadow: 0 0 10px var(--accent-glow); }
        button.ensemble { border: 1px solid var(--ensemble); color: var(--ensemble); grid-column: span 2; }
        button.ensemble:hover, button.ensemble.active { background: var(--ensemble); color: #000; }
        
        .util-row { display: flex; gap: 10px; margin-top: 10px; }
        .btn-util { flex: 1; background: #1e293b; border: 1px solid #475569; }

        /* RESULTS SECTION */
        .result-big {
            text-align: center; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;
            margin-bottom: 15px; display: flex; align-items: center; justify-content: center; gap: 20px;
        }
        .digit { font-size: 4rem; font-weight: bold; color: var(--accent); line-height: 1; }
        .meta { text-align: left; }
        .meta div { margin-bottom: 4px; font-size: 0.9rem; }
        .conf-high { color: var(--success); }
        .conf-low { color: #f87171; }

        /* PROBABILITY BARS */
        .prob-container { display: flex; flex-direction: column; gap: 4px; height: 200px; overflow-y: auto; }
        .prob-row { display: flex; align-items: center; gap: 10px; font-size: 0.8rem; }
        .prob-bar-bg { flex: 1; height: 6px; background: #334155; border-radius: 3px; overflow: hidden; }
        .prob-bar-fill { height: 100%; background: var(--dim); width: 0%; transition: width 0.3s; }
        .prob-bar-fill.winner { background: var(--accent); }

        /* BENCHMARK TABLE */
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th { text-align: left; color: var(--dim); padding: 8px; border-bottom: 1px solid #334155; }
        td { padding: 8px; border-bottom: 1px solid #334155; }
        tr:last-child td { border-bottom: none; }
        .row-ensemble { background: rgba(192, 132, 252, 0.1); }
        .row-ensemble td { color: var(--ensemble); font-weight: bold; }
        
        /* Scanline Anim */
        .scanline {
            position: absolute; width: 100%; height: 2px; background: var(--accent);
            opacity: 0.6; top: 0; left: 0; animation: scan 2s linear infinite; pointer-events: none;
        }
        @keyframes scan { 0% {top: 0;} 100% {top: 100%;} }
    </style>
</head>
<body>

    <div class="header">
        <h1>CNN ARCHITECT <span style="font-size:0.6em; color:var(--dim);">// MULTI-MODEL</span></h1>
        <div class="badge" id="sysStatus">SYSTEM IDLE</div>
    </div>

    <div class="grid-container">
        <div class="card">
            <div class="view-row">
                <div class="cam-box">
                    <div class="scanline"></div>
                    <img id="camImg" src="" alt="Live">
                    <div class="label">LIVE FEED</div>
                </div>
                <div class="proc-box">
                    <img id="procImg" src="" alt="Proc">
                    <div class="label">NN INPUT (32px)</div>
                </div>
            </div>

            <div style="margin-bottom:5px; font-size:0.8rem; color:var(--dim);">SELECT MODEL ARCHITECTURE:</div>
            <div class="model-grid">
                <button onclick="runInference(0)" id="btn0">SqueezeNet</button>
                <button onclick="runInference(1)" id="btn1">MobileNetV2</button>
                <button onclick="runInference(2)" id="btn2">ResNet-8</button>
                <button onclick="runInference(3)" id="btn3">EfficientNet</button>
                <button onclick="runInference(4)" id="btn4" class="ensemble">⚡ RUN ENSEMBLE FUSION</button>
            </div>
            
            <div class="util-row">
                <button class="btn-util" onclick="refreshCam()">Refresh Cam</button>
                <button class="btn-util" onclick="toggleFlash()">Toggle Flash</button>
            </div>
        </div>

        <div class="card">
            <div style="margin-bottom:10px; font-size:0.8rem; color:var(--dim);">INFERENCE RESULT:</div>
            
            <div class="result-big">
                <div class="digit" id="resDigit">-</div>
                <div class="meta">
                    <div id="resModel" style="color:var(--accent); font-weight:bold;">NO DATA</div>
                    <div id="resConf">Conf: --%</div>
                    <div id="resTime">Time: --ms</div>
                </div>
            </div>

            <div style="margin-bottom:5px; font-size:0.8rem; color:var(--dim);">SOFTMAX PROBABILITIES:</div>
            <div class="prob-container" id="probContainer">
                </div>
        </div>

        <div class="card full-width">
            <div style="margin-bottom:10px; font-size:0.8rem; color:var(--dim);">MODEL BENCHMARK:</div>
            <table>
                <thead>
                    <tr>
                        <th>Model Architecture</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Inference Time</th>
                    </tr>
                </thead>
                <tbody id="benchBody">
                    <tr><td colspan="4" style="text-align:center; color:var(--dim);">Run a prediction to see data</td></tr>
                </tbody>
            </table>
        </div>
    </div>

<script>
    const modelNames = ['SqueezeNet', 'MobileNetV2', 'ResNet-8', 'EfficientNet', 'Ensemble'];
    
    // Init Live View
    window.onload = function() {
        refreshCam();
        initProbBars();
    };

    function refreshCam() {
        document.getElementById('camImg').src = "/snapshot?t=" + new Date().getTime();
    }

    function initProbBars() {
        let html = '';
        for(let i=0; i<10; i++) {
            html += `
            <div class="prob-row">
                <div style="width:15px; text-align:center;">${i}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" id="bar${i}"></div>
                </div>
                <div style="width:35px; text-align:right;" id="val${i}">0%</div>
            </div>`;
        }
        document.getElementById('probContainer').innerHTML = html;
    }

    async function toggleFlash() {
        await fetch('/flash?on=' + (Math.random() > 0.5 ? 1 : 0)); // Simple toggle logic needs state in real app
        refreshCam();
    }

    async function runInference(id) {
        // UI Updates
        document.getElementById('sysStatus').innerText = "PROCESSING...";
        document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        document.getElementById('btn'+id).classList.add('active');

        try {
            const response = await fetch('/predict?model=' + id);
            const data = await response.json();

            // 1. Update Images
            document.getElementById('camImg').src = "/snapshot?t=" + new Date().getTime();
            document.getElementById('procImg').src = "/debug_input?t=" + new Date().getTime();

            // 2. Update Big Result
            document.getElementById('resDigit').innerText = data.prediction;
            document.getElementById('resModel').innerText = modelNames[id].toUpperCase();
            document.getElementById('resConf').innerText = "Conf: " + (data.confidence*100).toFixed(1) + "%";
            document.getElementById('resTime').innerText = "Time: " + data.time + "ms";
            
            const confEl = document.getElementById('resConf');
            confEl.className = data.confidence > 0.7 ? 'conf-high' : 'conf-low';

            // 3. Update Prob Bars
            for(let i=0; i<10; i++) {
                const p = data.probs[i] * 100;
                const bar = document.getElementById('bar'+i);
                bar.style.width = p + "%";
                bar.className = "prob-bar-fill" + (i === data.prediction ? " winner" : "");
                document.getElementById('val'+i).innerText = p.toFixed(1) + "%";
            }

            // 4. Update Benchmark Table (Simulate history or fetch all if available)
            // Since the backend 'predict' endpoint only returns the requested model's result (unless ensemble),
            // we will just update the row for the current model or all if ensemble.
            updateTable(data, id);

            document.getElementById('sysStatus').innerText = "IDLE";

        } catch (e) {
            console.error(e);
            document.getElementById('sysStatus').innerText = "ERROR";
        }
    }

    // Stores last known results for the table
    let benchData = [
        {name: 'SqueezeNet', pred: '-', conf: '-', time: '-'},
        {name: 'MobileNetV2', pred: '-', conf: '-', time: '-'},
        {name: 'ResNet-8', pred: '-', conf: '-', time: '-'},
        {name: 'EfficientNet', pred: '-', conf: '-', time: '-'},
        {name: 'Ensemble', pred: '-', conf: '-', time: '-'}
    ];

    function updateTable(data, currentId) {
        // If Ensemble (id=4), the backend usually runs ALL models. 
        // Note: The C++ code for handlePredict returns JSON. 
        // If we modify C++ to return "all" field for ensemble, we can parse it.
        // For now, let's update the current ID row.
        
        benchData[currentId] = {
            name: modelNames[currentId],
            pred: data.prediction,
            conf: (data.confidence * 100).toFixed(1) + '%',
            time: data.time + 'ms'
        };

        // Render Table
        let html = '';
        benchData.forEach((row, idx) => {
            const isEnsemble = idx === 4;
            html += `
            <tr class="${isEnsemble ? 'row-ensemble' : ''}">
                <td>${row.name}</td>
                <td>${row.pred}</td>
                <td>${row.conf}</td>
                <td>${row.time}</td>
            </tr>`;
        });
        document.getElementById('benchBody').innerHTML = html;
    }
</script>
</body>
</html>
)rawliteral";

// ====================== HTTP Handlers ======================
esp_err_t handleRoot(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, WEB_PAGE, strlen(WEB_PAGE));
}

esp_err_t handleSnapshot(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    // Convert to JPEG for web display
    size_t jpgLen = 0;
    uint8_t *jpgBuf = NULL;
    bool converted = frame2jpg(fb, 80, &jpgBuf, &jpgLen);
    esp_camera_fb_return(fb);
    
    if (!converted) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    esp_err_t res = httpd_resp_send(req, (const char*)jpgBuf, jpgLen);
    free(jpgBuf);
    return res;
}

esp_err_t handleDebugImg(httpd_req_t *req) {
    // Send 32x32 BMP of processed input
    const int w = IMG_INPUT_DIM, h = IMG_INPUT_DIM;
    const int hdrSize = 54 + 256 * 4;
    const int imgSize = w * h;
    const int fileSize = hdrSize + imgSize;
    
    uint8_t* bmp = (uint8_t*)malloc(fileSize);
    if (!bmp) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
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
    
    for (int i = 0; i < 256; i++) {
        bmp[54 + i * 4 + 0] = i;
        bmp[54 + i * 4 + 1] = i;
        bmp[54 + i * 4 + 2] = i;
    }
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bmp[hdrSize + (h - 1 - y) * w + x] = processedPreview[y * w + x];
        }
    }
    
    httpd_resp_set_type(req, "image/bmp");
    esp_err_t res = httpd_resp_send(req, (const char*)bmp, fileSize);
    free(bmp);
    return res;
}

esp_err_t handlePredict(httpd_req_t *req) {
    char query[32];
    int modelId = 0;
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[8];
        if (httpd_query_key_value(query, "model", val, sizeof(val)) == ESP_OK) {
            modelId = atoi(val);
        }
    }
    
    int result;
    if (modelId == MDL_ENSEMBLE) {
        result = runEnsemble();
    } else {
        result = runPrediction(modelId);
    }
    
    // Build JSON response
    char json[1024];
    char probStr[256] = "[";
    for (int i = 0; i < DIGIT_CLASSES; i++) {
        char tmp[16];
        sprintf(tmp, "%.4f%s", predictionProbs[modelId][i], i < 9 ? "," : "");
        strcat(probStr, tmp);
    }
    strcat(probStr, "]");
    
    sprintf(json, "{\"prediction\":%d,\"confidence\":%.4f,\"time\":%lu,\"probs\":%s}",
        predictedDigits[modelId], confidenceScores[modelId], inferenceMs[modelId], probStr);
    
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, json, strlen(json));
}

esp_err_t handleFlash(httpd_req_t *req) {
    char query[16];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[4];
        if (httpd_query_key_value(query, "on", val, sizeof(val)) == ESP_OK) {
            digitalWrite(LED_FLASH, atoi(val) ? HIGH : LOW);
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

esp_err_t handleReset(httpd_req_t *req) {
    // Clear cached predictions
    for (int m = 0; m < 5; m++) {
        predictedDigits[m] = -1;
        confidenceScores[m] = 0;
        for (int c = 0; c < DIGIT_CLASSES; c++) {
            predictionProbs[m][c] = 0;
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

void startWebServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 10;
    
    if (httpd_start(&webServer, &config) == ESP_OK) {
        httpd_uri_t uriRoot = { "/", HTTP_GET, handleRoot, NULL };
        httpd_uri_t uriSnap = { "/snapshot", HTTP_GET, handleSnapshot, NULL };
        httpd_uri_t uriDebug = { "/debug_input", HTTP_GET, handleDebugImg, NULL };
        httpd_uri_t uriPredict = { "/predict", HTTP_GET, handlePredict, NULL };
        httpd_uri_t uriFlash = { "/flash", HTTP_GET, handleFlash, NULL };
        httpd_uri_t uriReset = { "/reset", HTTP_GET, handleReset, NULL };
        
        httpd_register_uri_handler(webServer, &uriRoot);
        httpd_register_uri_handler(webServer, &uriSnap);
        httpd_register_uri_handler(webServer, &uriDebug);
        httpd_register_uri_handler(webServer, &uriPredict);
        httpd_register_uri_handler(webServer, &uriFlash);
        httpd_register_uri_handler(webServer, &uriReset);
        
        Serial.println("[Web] Server started");
    }
}

// ====================== Main Setup & Loop ======================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);  // Disable brownout
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n========================================");
    Serial.println("  EE4065 Q4: Multi-Model Digit Recognition");
    Serial.println("  Author: Yusuf");
    Serial.println("========================================\n");
    
    // Initialize LED
    pinMode(LED_FLASH, OUTPUT);
    digitalWrite(LED_FLASH, LOW);
    
    // Allocate PSRAM buffers
    if (psramFound()) {
        tensorMemory = (uint8_t*)ps_malloc(ARENA_BYTES);
        grayBuffer = (uint8_t*)ps_malloc(PREPROCESS_DIM * PREPROCESS_DIM);
        binaryBuffer = (uint8_t*)ps_malloc(PREPROCESS_DIM * PREPROCESS_DIM);
        morphBuffer = (uint8_t*)ps_malloc(PREPROCESS_DIM * PREPROCESS_DIM);
        Serial.printf("[Memory] PSRAM allocated: %d KB tensor + %d KB buffers\n", 
            ARENA_BYTES/1024, (PREPROCESS_DIM*PREPROCESS_DIM*3)/1024);
    } else {
        Serial.println("[Memory] ERROR: PSRAM not found!");
        while(1) delay(1000);
    }
    
    // Initialize camera
    if (!setupCamera()) {
        Serial.println("[Camera] FAILED - halting");
        while(1) delay(1000);
    }
    
    // Load initial model
    if (!setupTFLite(MDL_SQUEEZE)) {
        Serial.println("[TFLite] FAILED - halting");
        while(1) delay(1000);
    }
    
    // Connect to WiFi
    Serial.print("[WiFi] Connecting to ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    int wifiAttempts = 0;
    while (WiFi.status() != WL_CONNECTED && wifiAttempts++ < 30) {
        delay(500);
        Serial.print(".");
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected!");
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\n[WiFi] Failed - starting AP mode");
        WiFi.softAP(AP_SSID, AP_PASS);
        Serial.print("[WiFi] AP IP: ");
        Serial.println(WiFi.softAPIP());
    }
    
    // Start web server
    startWebServer();
    
    Serial.println("\n[System] Ready! Open browser to access web interface.");
}

void loop() {
    delay(10);
}
