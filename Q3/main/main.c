/**
 * Soru 3: ESP32 CAM için Upsampling ve Downsampling Modülü (ESP-IDF)
 * 
 * a- (10 points) perform upsampling with a given value
 * b- (10 points) perform downsampling with a given value
 * 
 * Tam sayı olmayan değerler için destek (1.5, 2/3, vb.)
 */

#include "esp_camera.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>
#include <stdint.h>

static const char *TAG = "UPSAMPLING_DOWNSAMPLING";

// Maksimum görüntü boyutları (hafıza kısıtları için)
#define MAX_WIDTH 640
#define MAX_HEIGHT 480

// Interpolation yöntemleri
typedef enum {
    INTERP_NEAREST,   // En yakın komşu
    INTERP_BILINEAR,  // Çift doğrusal interpolasyon
    INTERP_BICUBIC    // Çift kübik interpolasyon (daha yavaş ama daha iyi)
} interpolation_method_t;

/**
 * Bilinear interpolation - iki piksel arasında interpolasyon
 */
uint8_t bilinear_interpolate(uint8_t p00, uint8_t p01, uint8_t p10, uint8_t p11, 
                            float fx, float fy) {
    float a = p00 * (1.0f - fx) * (1.0f - fy);
    float b = p01 * fx * (1.0f - fy);
    float c = p10 * (1.0f - fx) * fy;
    float d = p11 * fx * fy;
    return (uint8_t)(a + b + c + d + 0.5f);
}

/**
 * Bicubic interpolation için yardımcı fonksiyon
 */
float cubic_weight(float x) {
    float abs_x = fabsf(x);
    if (abs_x <= 1.0f) {
        return 1.5f * abs_x * abs_x * abs_x - 2.5f * abs_x * abs_x + 1.0f;
    } else if (abs_x <= 2.0f) {
        return -0.5f * abs_x * abs_x * abs_x + 2.5f * abs_x * abs_x - 4.0f * abs_x + 2.0f;
    }
    return 0.0f;
}

/**
 * Bicubic interpolation - 4x4 piksel alanından interpolasyon
 */
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

/**
 * RGB565'ten gri tonlamaya çevir
 */
uint8_t rgb565_to_gray(uint16_t pixel) {
    uint8_t r = ((pixel >> 11) & 0x1F) << 3;
    uint8_t g = ((pixel >> 5) & 0x3F) << 2;
    uint8_t b = (pixel & 0x1F) << 3;
    return (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
}

/**
 * Gri tonlamadan RGB565'e çevir
 */
uint16_t gray_to_rgb565(uint8_t gray) {
    uint8_t r = gray;
    uint8_t g = gray;
    uint8_t b = gray;
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

/**
 * Upsampling - Görüntüyü büyütme
 */
void upsampling(uint16_t *input_image, int input_width, int input_height,
                uint16_t *output_image, int *output_width, int *output_height,
                float scale_factor, interpolation_method_t method) {
    
    // Çıkış boyutlarını hesapla
    *output_width = (int)(input_width * scale_factor + 0.5f);
    *output_height = (int)(input_height * scale_factor + 0.5f);
    
    // Boyut kontrolü
    if (*output_width > MAX_WIDTH || *output_height > MAX_HEIGHT) {
        ESP_LOGE(TAG, "HATA: Çıkış boyutu çok büyük!");
        return;
    }
    
    ESP_LOGI(TAG, "Upsampling: %dx%d -> %dx%d (scale: %.2f)", 
             input_width, input_height, *output_width, *output_height, scale_factor);
    
    // Ölçek faktörlerini hesapla
    float scale_x = (float)input_width / *output_width;
    float scale_y = (float)input_height / *output_height;
    
    if (method == INTERP_NEAREST) {
        // En yakın komşu interpolasyonu (en hızlı)
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
        // Çift doğrusal interpolasyon (dengeli hız/kalite)
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
        // Çift kübik interpolasyon (en yavaş ama en iyi kalite)
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

/**
 * Downsampling - Görüntüyü küçültme
 */
void downsampling(uint16_t *input_image, int input_width, int input_height,
                  uint16_t *output_image, int *output_width, int *output_height,
                  float scale_factor, interpolation_method_t method) {
    
    // Çıkış boyutlarını hesapla
    *output_width = (int)(input_width * scale_factor + 0.5f);
    *output_height = (int)(input_height * scale_factor + 0.5f);
    
    // Minimum boyut kontrolü
    if (*output_width < 1) *output_width = 1;
    if (*output_height < 1) *output_height = 1;
    
    ESP_LOGI(TAG, "Downsampling: %dx%d -> %dx%d (scale: %.2f)", 
             input_width, input_height, *output_width, *output_height, scale_factor);
    
    // Ölçek faktörlerini hesapla
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

void app_main(void) {
    ESP_LOGI(TAG, "ESP32 CAM Upsampling/Downsampling Modülü Başlatılıyor...");
    
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
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 1;
    
    // Kamera başlat
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Kamera başlatma hatası: %s", esp_err_to_name(err));
        return;
    }
    
    ESP_LOGI(TAG, "Kamera başlatıldı");
    vTaskDelay(1000 / portTICK_PERIOD_MS);
    
    // Statik bellek tahsisi
    static uint16_t upsampled_image[MAX_WIDTH * MAX_HEIGHT];
    static uint16_t downsampled_image[MAX_WIDTH * MAX_HEIGHT];
    
    // Ana döngü
    while (1) {
        // Frame yakala
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Frame yakalama hatası");
            vTaskDelay(1000 / portTICK_PERIOD_MS);
            continue;
        }
        
        int input_width = fb->width;
        int input_height = fb->height;
        
        ESP_LOGI(TAG, "\n=== Upsampling/Downsampling Test ===");
        ESP_LOGI(TAG, "Giriş görüntü boyutu: %dx%d", input_width, input_height);
        
        int upsampled_width, upsampled_height;
        int downsampled_width, downsampled_height;
        
        // Test 1: Upsampling (1.5x - tam sayı olmayan)
        ESP_LOGI(TAG, "\n--- Test 1: Upsampling 1.5x (Bilinear) ---");
        float upscale = 1.5f;
        upsampling((uint16_t *)fb->buf, input_width, input_height,
                   upsampled_image, &upsampled_width, &upsampled_height,
                   upscale, INTERP_BILINEAR);
        ESP_LOGI(TAG, "Sonuç: %dx%d", upsampled_width, upsampled_height);
        
        // Test 2: Downsampling (2/3 = 0.667 - tam sayı olmayan)
        ESP_LOGI(TAG, "\n--- Test 2: Downsampling 2/3 (Bilinear) ---");
        float downscale = 2.0f / 3.0f;  // 0.667
        downsampling((uint16_t *)fb->buf, input_width, input_height,
                     downsampled_image, &downsampled_width, &downsampled_height,
                     downscale, INTERP_BILINEAR);
        ESP_LOGI(TAG, "Sonuç: %dx%d", downsampled_width, downsampled_height);
        
        // Test 3: Upsampling (2.0x - tam sayı)
        ESP_LOGI(TAG, "\n--- Test 3: Upsampling 2.0x (Bicubic) ---");
        upsampling((uint16_t *)fb->buf, input_width, input_height,
                   upsampled_image, &upsampled_width, &upsampled_height,
                   2.0f, INTERP_BICUBIC);
        ESP_LOGI(TAG, "Sonuç: %dx%d", upsampled_width, upsampled_height);
        
        // Test 4: Downsampling (0.5x - tam sayı)
        ESP_LOGI(TAG, "\n--- Test 4: Downsampling 0.5x (Nearest) ---");
        downsampling((uint16_t *)fb->buf, input_width, input_height,
                     downsampled_image, &downsampled_width, &downsampled_height,
                     0.5f, INTERP_NEAREST);
        ESP_LOGI(TAG, "Sonuç: %dx%d", downsampled_width, downsampled_height);
        
        // Test 5: Upsampling (1.25x - tam sayı olmayan)
        ESP_LOGI(TAG, "\n--- Test 5: Upsampling 1.25x (Bilinear) ---");
        upsampling((uint16_t *)fb->buf, input_width, input_height,
                   upsampled_image, &upsampled_width, &upsampled_height,
                   1.25f, INTERP_BILINEAR);
        ESP_LOGI(TAG, "Sonuç: %dx%d", upsampled_width, upsampled_height);
        
        // Hafıza bilgisi
        ESP_LOGI(TAG, "\nSerbest heap: %d bytes", esp_get_free_heap_size());
        
        // Frame'i serbest bırak
        esp_camera_fb_return(fb);
        
        // 5 saniye bekle
        vTaskDelay(5000 / portTICK_PERIOD_MS);
    }
}
