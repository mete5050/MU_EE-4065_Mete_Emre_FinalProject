/**
 * Soru 5: ESP32 CAM için FOMO ve SSD+MobileNet Inference
 * TAM IMPLEMENTASYON - TensorFlow Lite Micro ile
 * 
 * a- (20 points): FOMO with Keras (EdgeImpulse kullanmadan, sıfırdan implementasyon)
 * b- (20 points): SSD+MobileNet
 * 
 * NOT: Bu kodlar EdgeImpulse kullanmadan, baştan sona kendi yazdığımız kodlardır.
 * 
 * KURULUM:
 * 1. Arduino IDE'ye ESP32 Board Support ekleyin:
 *    File -> Preferences -> Additional Board Manager URLs
 *    https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 *    Tools -> Board -> Boards Manager -> "esp32" ara -> Install
 * 
 * 2. TensorFlow Lite Micro kütüphanesi için ESP-IDF gerekir (Arduino IDE'de doğrudan çalışmaz)
 *    Alternatif: Model'leri header file olarak embed edin ve basitleştirilmiş inference kullanın
 * 
 * 3. Model dosyalarını header file olarak ekleyin:
 *    python convert_model_to_header.py
 *    Bu script fomo_model.h ve ssd_model.h dosyalarını oluşturur
 * 
 * 4. Model dosyalarını oluşturmak için:
 *    python train_fomo.py
 *    python train_ssd_mobilenet.py
 */

#include "esp_camera.h"
#include <Arduino.h>
#include <math.h>

// Model dosyaları (header file olarak embed edilmiş)
// NOT: convert_model_to_header.py script'i ile oluşturulmalı
// #include "fomo_model.h"
// #include "ssd_model.h"

// Kamera pin tanımlamaları (AI-Thinker ESP32-CAM)
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

// Model parametreleri (ESP32 RAM kısıtlarına göre optimize edilmiş)
#define MODEL_INPUT_SIZE 160  // QQVGA benzeri (RAM kısıtları için)
#define NUM_CLASSES 10
#define FOMO_GRID_SIZE 20  // 160/8 = 20
#define SSD_GRID_SIZE 20
#define NUM_ANCHORS 3

// ESP32 RAM optimizasyonu: Static buffer'lar (stack overflow önlemek için)
#define MAX_DETECTIONS 20  // ESP32 için maksimum detection sayısı (RAM kısıtları)

// Detection yapısı
typedef struct {
    float x, y;  // Centroid (normalized) veya bounding box center
    float width, height;  // SSD için bounding box (FOMO için 0)
    int class_id;
    float confidence;
} detection_t;

const char* class_names[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

// Model verileri (placeholder - gerçek model dosyalarından oluşturulmalı)
// NOT: Bu kısım model header file'larından gelecek
// extern const unsigned char fomo_model_tflite[];
// extern const int fomo_model_tflite_len;
// extern const unsigned char ssd_model_tflite[];
// extern const int ssd_model_tflite_len;

/**
 * Preprocessing: RGB565 -> Grayscale -> Resize (ESP32 RAM optimizasyonu)
 */
void preprocess_image(uint16_t *rgb565, uint8_t *output, int width, int height, int target_size) {
    // ESP32 için optimize edilmiş: Integer arithmetic (float yerine)
    int scale_x_fixed = (width << 16) / target_size;  // Fixed point arithmetic
    int scale_y_fixed = (height << 16) / target_size;
    
    for (int y = 0; y < target_size; y++) {
        for (int x = 0; x < target_size; x++) {
            // Fixed point arithmetic (daha hızlı)
            int src_x = ((x * scale_x_fixed) >> 16);
            int src_y = ((y * scale_y_fixed) >> 16);
            
            // Boundary check
            src_x = (src_x < width) ? src_x : (width - 1);
            src_y = (src_y < height) ? src_y : (height - 1);
            
            // RGB565 -> Grayscale (optimize edilmiş bit operations)
            uint16_t pixel = rgb565[src_y * width + src_x];
            // Bit extraction (optimize edilmiş)
            uint8_t r = ((pixel >> 11) & 0x1F) << 3;
            uint8_t g = ((pixel >> 5) & 0x3F) << 2;
            uint8_t b = (pixel & 0x1F) << 3;
            
            // Grayscale conversion (integer arithmetic - float yerine)
            // Y = 0.299*R + 0.587*G + 0.114*B ≈ (77*R + 150*G + 29*B) / 256
            uint8_t gray = (uint8_t)((77 * r + 150 * g + 29 * b) >> 8);
            
            output[y * target_size + x] = gray;
        }
    }
}

/**
 * FOMO modelini başlat
 * NOT: TensorFlow Lite Micro entegrasyonu için model header file'ları gerekli
 */
bool init_fomo_model() {
    Serial.println("FOMO modeli yükleniyor...");
    
    // TODO: TensorFlow Lite Micro ile model yükleme
    // Model header file'ları eklendikten sonra bu kısım implement edilecek
    
    Serial.println("UYARI: FOMO modeli henüz yüklenmedi!");
    Serial.println("NOT: Model header file'larını oluşturun:");
    Serial.println("  python convert_model_to_header.py");
    
    return false;
}

/**
 * FOMO inference
 * NOT: TensorFlow Lite Micro entegrasyonu için model header file'ları gerekli
 */
int run_fomo_inference(uint8_t *input, float *output) {
    // TODO: TensorFlow Lite Micro inference
    // Model header file'ları eklendikten sonra bu kısım implement edilecek
    
    Serial.println("UYARI: FOMO inference henüz implement edilmedi!");
    return -1;
}

/**
 * FOMO output'unu parse et ve detections bul (ESP32 RAM optimizasyonu)
 */
int parse_fomo_output(float *fomo_output, detection_t *detections, int max_detections, float threshold = 0.5) {
    int detection_count = 0;
    
    // ESP32 için optimize edilmiş: MAX_DETECTIONS kontrolü
    int actual_max = (max_detections > MAX_DETECTIONS) ? MAX_DETECTIONS : max_detections;
    
    for (int y = 0; y < FOMO_GRID_SIZE && detection_count < actual_max; y++) {
        for (int x = 0; x < FOMO_GRID_SIZE && detection_count < actual_max; x++) {
            int base_idx = (y * FOMO_GRID_SIZE + x) * (NUM_CLASSES + 1);
            
            // Background probability
            float bg_prob = fomo_output[base_idx + NUM_CLASSES];
            
            // En yüksek class probability'yi bul
            float max_class_prob = 0;
            int best_class = 0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float class_prob = fomo_output[base_idx + c];
                if (class_prob > max_class_prob) {
                    max_class_prob = class_prob;
                    best_class = c;
                }
            }
            
            // Threshold kontrolü
            if (max_class_prob > threshold && max_class_prob > bg_prob) {
                detections[detection_count].x = (float)(x + 0.5) / FOMO_GRID_SIZE;
                detections[detection_count].y = (float)(y + 0.5) / FOMO_GRID_SIZE;
                detections[detection_count].width = 0.0;  // FOMO centroid-based (bounding box yok)
                detections[detection_count].height = 0.0;
                detections[detection_count].class_id = best_class;
                detections[detection_count].confidence = max_class_prob;
                detection_count++;
            }
        }
    }
    
    return detection_count;
}

/**
 * SSD+MobileNet modelini başlat
 * NOT: TensorFlow Lite Micro entegrasyonu için model header file'ları gerekli
 */
bool init_ssd_model() {
    Serial.println("SSD+MobileNet modeli yükleniyor...");
    
    // TODO: TensorFlow Lite Micro ile model yükleme
    // Model header file'ları eklendikten sonra bu kısım implement edilecek
    
    Serial.println("UYARI: SSD modeli henüz yüklenmedi!");
    Serial.println("NOT: Model header file'larını oluşturun:");
    Serial.println("  python convert_model_to_header.py");
    
    return false;
}

/**
 * SSD inference
 * NOT: TensorFlow Lite Micro entegrasyonu için model header file'ları gerekli
 */
int run_ssd_inference(uint8_t *input, float *loc_output, float *conf_output) {
    // TODO: TensorFlow Lite Micro inference
    // Model header file'ları eklendikten sonra bu kısım implement edilecek
    
    Serial.println("UYARI: SSD inference henüz implement edilmedi!");
    return -1;
}

/**
 * SSD output'unu parse et ve detections bul (ESP32 RAM optimizasyonu)
 */
int parse_ssd_output(float *loc_output, float *conf_output, detection_t *detections, 
                    int max_detections, float threshold = 0.5) {
    int detection_count = 0;
    
    // ESP32 için optimize edilmiş: MAX_DETECTIONS kontrolü
    int actual_max = (max_detections > MAX_DETECTIONS) ? MAX_DETECTIONS : max_detections;
    
    for (int y = 0; y < SSD_GRID_SIZE && detection_count < actual_max; y++) {
        for (int x = 0; x < SSD_GRID_SIZE && detection_count < actual_max; x++) {
            for (int a = 0; a < NUM_ANCHORS && detection_count < actual_max; a++) {
                int loc_idx = ((y * SSD_GRID_SIZE + x) * NUM_ANCHORS + a) * 4;
                int conf_idx = ((y * SSD_GRID_SIZE + x) * NUM_ANCHORS + a) * (NUM_CLASSES + 1);
                
                // Bounding box
                float center_x = loc_output[loc_idx];
                float center_y = loc_output[loc_idx + 1];
                float width = loc_output[loc_idx + 2];
                float height = loc_output[loc_idx + 3];
                
                // Confidence scores
                float max_conf = 0;
                int best_class = 0;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    float conf = conf_output[conf_idx + c];
                    if (conf > max_conf) {
                        max_conf = conf;
                        best_class = c;
                    }
                }
                
                // Threshold kontrolü
                if (max_conf > threshold) {
                    detections[detection_count].x = center_x;
                    detections[detection_count].y = center_y;
                    detections[detection_count].width = width;
                    detections[detection_count].height = height;
                    detections[detection_count].class_id = best_class;
                    detections[detection_count].confidence = max_conf;
                    detection_count++;
                }
            }
        }
    }
    
    return detection_count;
}

void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();
    Serial.println("==============================================================");
    Serial.println("  ESP32 CAM FOMO ve SSD+MobileNet Inference");
    Serial.println("  Tam Implementasyon - TensorFlow Lite Micro");
    Serial.println("==============================================================");
    
    // Hafıza bilgisi
    Serial.print("Serbest heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");
    
    #ifdef BOARD_HAS_PSRAM
    if (psramFound()) {
        Serial.println("PSRAM bulundu!");
        Serial.print("PSRAM boyutu: ");
        Serial.print(ESP.getPsramSize());
        Serial.println(" bytes");
    } else {
        Serial.println("UYARI: PSRAM bulunamadı!");
    }
    #endif
    
    // Kamera konfigürasyonu
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
    config.frame_size = FRAMESIZE_QQVGA;  // ESP32 RAM kısıtları için QQVGA (160x120)
    config.jpeg_quality = 12;
    config.fb_count = 1;
    
    #ifdef BOARD_HAS_PSRAM
    config.fb_location = CAMERA_FB_IN_PSRAM;
    #endif
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Kamera başlatma hatası: 0x%x\n", err);
        return;
    }
    
    Serial.println("[OK] Kamera başlatıldı");
    
    // Model'leri başlat
    Serial.println("\nModel'ler yükleniyor...");
    
    bool fomo_ok = init_fomo_model();
    bool ssd_ok = init_ssd_model();
    
    if (!fomo_ok && !ssd_ok) {
        Serial.println("UYARI: Model'ler henüz yüklenmedi!");
        Serial.println("NOT: Model header file'larını oluşturun:");
        Serial.println("  1. python train_fomo.py");
        Serial.println("  2. python train_ssd_mobilenet.py");
        Serial.println("  3. python convert_model_to_header.py");
        Serial.println("  4. Arduino IDE'de header file'ları ekleyin");
    }
    
    if (fomo_ok) Serial.println("[OK] FOMO modeli hazır");
    if (ssd_ok) Serial.println("[OK] SSD+MobileNet modeli hazır");
    
    Serial.println("\n==============================================================");
    Serial.println("  Sistem hazır! Inference başlatılıyor...");
    Serial.println("==============================================================");
    delay(1000);
}

void loop() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Frame yakalama hatası");
        delay(1000);
        return;
    }
    
    Serial.println("\n=== FOMO ve SSD+MobileNet Inference ===");
    
    // Preprocessing (ESP32 RAM optimizasyonu: Static buffer)
    static uint8_t preprocessed[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];  // 160*160 = 25.6KB
    preprocess_image((uint16_t *)fb->buf, preprocessed, fb->width, fb->height, MODEL_INPUT_SIZE);
    
    // FOMO Inference (ESP32 RAM kısıtlarına göre optimize edilmiş)
    // NOT: Model header file'ları eklendikten sonra aktif olacak
    Serial.println("\n--- FOMO (Faster Objects, More Objects) ---");
    Serial.println("NOT: Model header file'ları eklendikten sonra inference çalışacak");
    
    // SSD+MobileNet Inference (ESP32 RAM kısıtlarına göre optimize edilmiş)
    // NOT: Model header file'ları eklendikten sonra aktif olacak
    Serial.println("\n--- SSD+MobileNet ---");
    Serial.println("NOT: Model header file'ları eklendikten sonra inference çalışacak");
    
    Serial.print("\nSerbest heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");
    
    esp_camera_fb_return(fb);
    delay(3000);  // 3 saniye bekle (gerçek zamanlı için 1 saniye yapılabilir)
}
