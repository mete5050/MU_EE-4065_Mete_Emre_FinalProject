/**
 * Soru 3: ESP32 CAM için Upsampling ve Downsampling Modülü
 * 
 * Assignment Requirements:
 * - a- (10 points) perform upsampling with a given value
 * - b- (10 points) perform downsampling with a given value
 * - Support for non-integer scale factors (1.5, 2/3, etc.)
 */

#include "esp_camera.h"
#include <Arduino.h>
#include <math.h>

// ============================================================================
// CAMERA PIN DEFINITIONS
// ============================================================================

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

// Maksimum görüntü boyutları (hafıza kısıtları için küçültüldü)
#define MAX_WIDTH 160
#define MAX_HEIGHT 120

// Interpolation yöntemleri
typedef enum {
    INTERP_NEAREST,
    INTERP_BILINEAR,
    INTERP_BICUBIC
} interpolation_method_t;

// ============================================================================
// INTERPOLATION FUNCTIONS
// ============================================================================

uint8_t bilinear_interpolate(uint8_t p00, uint8_t p01, uint8_t p10, uint8_t p11, 
                            float fx, float fy) {
    float a = p00 * (1.0f - fx) * (1.0f - fy);
    float b = p01 * fx * (1.0f - fy);
    float c = p10 * (1.0f - fx) * fy;
    float d = p11 * fx * fy;
    return (uint8_t)(a + b + c + d + 0.5f);
}

float cubic_weight(float x) {
    float abs_x = fabsf(x);
    if (abs_x <= 1.0f) {
        return 1.5f * abs_x * abs_x * abs_x - 2.5f * abs_x * abs_x + 1.0f;
    } else if (abs_x <= 2.0f) {
        return -0.5f * abs_x * abs_x * abs_x + 2.5f * abs_x * abs_x - 4.0f * abs_x + 2.0f;
    }
    return 0.0f;
}

uint8_t bicubic_interpolate(uint8_t pixels[4][4], float fx, float fy) {
    float result = 0.0f;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            float weight = cubic_weight(i - 1.0f - fx) * cubic_weight(j - 1.0f - fy);
            result += pixels[j][i] * weight;
        }
    }
    return (uint8_t)(fmaxf(0.0f, fminf(255.0f, result + 0.5f)));
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

uint8_t rgb565_to_gray(uint16_t pixel) {
    uint8_t r = ((pixel >> 11) & 0x1F) << 3;
    uint8_t g = ((pixel >> 5) & 0x3F) << 2;
    uint8_t b = (pixel & 0x1F) << 3;
    return (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
}

uint16_t gray_to_rgb565(uint8_t gray) {
    uint8_t r = gray;
    uint8_t g = gray;
    uint8_t b = gray;
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

// ============================================================================
// UPSAMPLING AND DOWNSAMPLING FUNCTIONS
// ============================================================================

void upsampling(uint16_t *input_image, int input_width, int input_height,
                uint16_t *output_image, int *output_width, int *output_height,
                float scale_factor, interpolation_method_t method) {
    
    *output_width = (int)(input_width * scale_factor + 0.5f);
    *output_height = (int)(input_height * scale_factor + 0.5f);
    
    if (*output_width > MAX_WIDTH || *output_height > MAX_HEIGHT) {
        Serial.println("[HATA] Cikis boyutu cok buyuk!");
        return;
    }
    
    float scale_x = (float)input_width / *output_width;
    float scale_y = (float)input_height / *output_height;
    
    if (method == INTERP_NEAREST) {
        for (int y = 0; y < *output_height; y++) {
            for (int x = 0; x < *output_width; x++) {
                int src_x = (int)(x * scale_x + 0.5f);
                int src_y = (int)(y * scale_y + 0.5f);
                src_x = (src_x < input_width) ? src_x : (input_width - 1);
                src_y = (src_y < input_height) ? src_y : (input_height - 1);
                output_image[y * (*output_width) + x] = 
                    input_image[src_y * input_width + src_x];
            }
        }
    }
    else if (method == INTERP_BILINEAR) {
        for (int y = 0; y < *output_height; y++) {
            for (int x = 0; x < *output_width; x++) {
                float src_x_f = x * scale_x;
                float src_y_f = y * scale_y;
                int src_x = (int)src_x_f;
                int src_y = (int)src_y_f;
                float fx = src_x_f - src_x;
                float fy = src_y_f - src_y;
                
                int x0 = (src_x < input_width) ? src_x : (input_width - 1);
                int y0 = (src_y < input_height) ? src_y : (input_height - 1);
                int x1 = (src_x + 1 < input_width) ? (src_x + 1) : (input_width - 1);
                int y1 = (src_y + 1 < input_height) ? (src_y + 1) : (input_height - 1);
                
                uint8_t p00 = rgb565_to_gray(input_image[y0 * input_width + x0]);
                uint8_t p01 = rgb565_to_gray(input_image[y0 * input_width + x1]);
                uint8_t p10 = rgb565_to_gray(input_image[y1 * input_width + x0]);
                uint8_t p11 = rgb565_to_gray(input_image[y1 * input_width + x1]);
                
                uint8_t interpolated = bilinear_interpolate(p00, p01, p10, p11, fx, fy);
                output_image[y * (*output_width) + x] = gray_to_rgb565(interpolated);
            }
        }
    }
    else if (method == INTERP_BICUBIC) {
        for (int y = 0; y < *output_height; y++) {
            for (int x = 0; x < *output_width; x++) {
                float src_x_f = x * scale_x;
                float src_y_f = y * scale_y;
                int src_x = (int)src_x_f;
                int src_y = (int)src_y_f;
                float fx = src_x_f - src_x;
                float fy = src_y_f - src_y;
                
                uint8_t pixels[4][4];
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        int px = src_x + i - 1;
                        int py = src_y + j - 1;
                        if (px < 0) px = 0;
                        if (px >= input_width) px = input_width - 1;
                        if (py < 0) py = 0;
                        if (py >= input_height) py = input_height - 1;
                        pixels[j][i] = rgb565_to_gray(input_image[py * input_width + px]);
                    }
                }
                
                uint8_t interpolated = bicubic_interpolate(pixels, fx, fy);
                output_image[y * (*output_width) + x] = gray_to_rgb565(interpolated);
            }
        }
    }
}

void downsampling(uint16_t *input_image, int input_width, int input_height,
                  uint16_t *output_image, int *output_width, int *output_height,
                  float scale_factor, interpolation_method_t method) {
    
    *output_width = (int)(input_width * scale_factor + 0.5f);
    *output_height = (int)(input_height * scale_factor + 0.5f);
    
    if (*output_width < 1) *output_width = 1;
    if (*output_height < 1) *output_height = 1;
    
    float scale_x = (float)input_width / *output_width;
    float scale_y = (float)input_height / *output_height;
    
    if (method == INTERP_NEAREST) {
        for (int y = 0; y < *output_height; y++) {
            for (int x = 0; x < *output_width; x++) {
                int src_x = (int)(x * scale_x + 0.5f);
                int src_y = (int)(y * scale_y + 0.5f);
                src_x = (src_x < input_width) ? src_x : (input_width - 1);
                src_y = (src_y < input_height) ? src_y : (input_height - 1);
                output_image[y * (*output_width) + x] = 
                    input_image[src_y * input_width + src_x];
            }
        }
    }
    else if (method == INTERP_BILINEAR) {
        for (int y = 0; y < *output_height; y++) {
            for (int x = 0; x < *output_width; x++) {
                float src_x_f = x * scale_x;
                float src_y_f = y * scale_y;
                int src_x = (int)src_x_f;
                int src_y = (int)src_y_f;
                float fx = src_x_f - src_x;
                float fy = src_y_f - src_y;
                
                int x0 = (src_x < input_width) ? src_x : (input_width - 1);
                int y0 = (src_y < input_height) ? src_y : (input_height - 1);
                int x1 = (src_x + 1 < input_width) ? (src_x + 1) : (input_width - 1);
                int y1 = (src_y + 1 < input_height) ? (src_y + 1) : (input_height - 1);
                
                uint8_t p00 = rgb565_to_gray(input_image[y0 * input_width + x0]);
                uint8_t p01 = rgb565_to_gray(input_image[y0 * input_width + x1]);
                uint8_t p10 = rgb565_to_gray(input_image[y1 * input_width + x0]);
                uint8_t p11 = rgb565_to_gray(input_image[y1 * input_width + x1]);
                
                uint8_t interpolated = bilinear_interpolate(p00, p01, p10, p11, fx, fy);
                output_image[y * (*output_width) + x] = gray_to_rgb565(interpolated);
            }
        }
    }
    else if (method == INTERP_BICUBIC) {
        for (int y = 0; y < *output_height; y++) {
            for (int x = 0; x < *output_width; x++) {
                float src_x_f = x * scale_x;
                float src_y_f = y * scale_y;
                int src_x = (int)src_x_f;
                int src_y = (int)src_y_f;
                float fx = src_x_f - src_x;
                float fy = src_y_f - src_y;
                
                uint8_t pixels[4][4];
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        int px = src_x + i - 1;
                        int py = src_y + j - 1;
                        if (px < 0) px = 0;
                        if (px >= input_width) px = input_width - 1;
                        if (py < 0) py = 0;
                        if (py >= input_height) py = input_height - 1;
                        pixels[j][i] = rgb565_to_gray(input_image[py * input_width + px]);
                    }
                }
                
                uint8_t interpolated = bicubic_interpolate(pixels, fx, fy);
                output_image[y * (*output_width) + x] = gray_to_rgb565(interpolated);
            }
        }
    }
}

// ============================================================================
// MAIN FUNCTIONS
// ============================================================================

void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(false);
    Serial.println();
    Serial.println("==============================================================");
    Serial.println("     ESP32 CAM UPSAMPLING/DOWNSAMPLING MODULU");
    Serial.println("==============================================================");
    
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
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[HATA] Kamera baslatma hatasi: 0x%x\n", err);
        return;
    }
    
    Serial.println("[OK] Kamera baslatildi");
    delay(1000);
}

void loop() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[HATA] Frame yakalama hatasi");
        delay(1000);
        return;
    }
    
    int input_width = fb->width;
    int input_height = fb->height;
    
    Serial.println("\n==============================================================");
    Serial.println("     UPSAMPLING VE DOWNSAMPLING TEST");
    Serial.println("==============================================================");
    Serial.print("Giris goruntu boyutu: ");
    Serial.print(input_width);
    Serial.print("x");
    Serial.println(input_height);
    
    static uint16_t processed_image[MAX_WIDTH * MAX_HEIGHT];
    
    int upsampled_width, upsampled_height;
    int downsampled_width, downsampled_height;
    
    // Upsampling 1.5x
    Serial.println("\n--------------------------------------------------------------");
    Serial.println("UPSAMPLING (1.5x) - BILINEAR");
    Serial.println("--------------------------------------------------------------");
    Serial.print("  Orijinal boyut: ");
    Serial.print(input_width);
    Serial.print("x");
    Serial.println(input_height);
    
    upsampling((uint16_t *)fb->buf, input_width, input_height,
               processed_image, &upsampled_width, &upsampled_height,
               1.5f, INTERP_BILINEAR);
    
    Serial.print("  Yeni boyut:    ");
    Serial.print(upsampled_width);
    Serial.print("x");
    Serial.println(upsampled_height);
    Serial.println("  [OK] Upsampling tamamlandi");
    
    // Downsampling 0.6x
    Serial.println("\n--------------------------------------------------------------");
    Serial.println("DOWNSAMPLING (0.6x) - BILINEAR");
    Serial.println("--------------------------------------------------------------");
    Serial.print("  Orijinal boyut: ");
    Serial.print(input_width);
    Serial.print("x");
    Serial.println(input_height);
    
    downsampling((uint16_t *)fb->buf, input_width, input_height,
                 processed_image, &downsampled_width, &downsampled_height,
                 0.6f, INTERP_BILINEAR);
    
    Serial.print("  Yeni boyut:    ");
    Serial.print(downsampled_width);
    Serial.print("x");
    Serial.println(downsampled_height);
    Serial.println("  [OK] Downsampling tamamlandi");
    
    // Özet
    Serial.println("\n==============================================================");
    Serial.println("     [OK] ISLEM TAMAMLANDI!");
    Serial.println("==============================================================");
    Serial.print("Serbest heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");
    Serial.println("==============================================================\n");
    
    esp_camera_fb_return(fb);
    delay(5000);
}
