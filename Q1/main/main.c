/**
 * Soru 1b: ESP32 CAM üzerinde çalışacak C thresholding kodu (ESP-IDF)
 * - Görüntüde parlak bir nesne var
 * - Arka plan pikselleri nesne piksellerinden daha koyu
 * - 1000 piksel boyutunda nesneyi tespit et
 */

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

static const char *TAG = "THRESHOLDING";

// Threshold değeri (Otsu veya manuel)
#define THRESHOLD_VALUE 127
#define TARGET_OBJECT_SIZE 1000
#define SIZE_TOLERANCE 50

// Görüntü boyutları
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480

// Connected components için etiketleme
#define MAX_LABELS 1000
#define BACKGROUND 0

// Blob yapısı
typedef struct {
    int area;
    int min_x, min_y, max_x, max_y;
    int centroid_x, centroid_y;
} blob_t;

// Otsu threshold hesaplama
int calculate_otsu_threshold(uint8_t *gray_image, int width, int height) {
    int histogram[256] = {0};
    int total_pixels = width * height;
    
    // Histogram oluştur
    for (int i = 0; i < total_pixels; i++) {
        histogram[gray_image[i]]++;
    }
    
    // Otsu algoritması
    float sum = 0;
    float sumB = 0;
    int q1 = 0;
    int q2 = 0;
    float var_max = 0;
    int threshold = 0;
    
    for (int i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }
    
    for (int t = 0; t < 256; t++) {
        q1 += histogram[t];
        if (q1 == 0) continue;
        
        q2 = total_pixels - q1;
        if (q2 == 0) break;
        
        sumB += t * histogram[t];
        float m1 = sumB / q1;
        float m2 = (sum - sumB) / q2;
        float var_between = (float)q1 * (float)q2 * (m1 - m2) * (m1 - m2);
        
        if (var_between > var_max) {
            var_max = var_between;
            threshold = t;
        }
    }
    
    return threshold;
}

// RGB565'ten RGB888'e dönüştür
void rgb565_to_rgb888(uint16_t *rgb565_image, uint8_t *rgb888_image, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        uint16_t pixel = rgb565_image[i];
        // RGB565 formatı: RRRRR GGGGGG BBBBB
        uint8_t r = ((pixel >> 11) & 0x1F) << 3;
        uint8_t g = ((pixel >> 5) & 0x3F) << 2;
        uint8_t b = (pixel & 0x1F) << 3;
        
        rgb888_image[i * 3] = r;
        rgb888_image[i * 3 + 1] = g;
        rgb888_image[i * 3 + 2] = b;
    }
}

// RGB'yi gri tonlamaya çevir
void rgb_to_gray(uint8_t *rgb_image, uint8_t *gray_image, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int r = rgb_image[i * 3];
        int g = rgb_image[i * 3 + 1];
        int b = rgb_image[i * 3 + 2];
        // Luminance formülü
        gray_image[i] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

// Thresholding uygula
void apply_threshold(uint8_t *gray_image, uint8_t *binary_image, int width, int height, int threshold) {
    for (int i = 0; i < width * height; i++) {
        binary_image[i] = (gray_image[i] > threshold) ? 255 : 0;
    }
}

// Connected components labeling (basit 4-bağlantılı)
int label_components(uint8_t *binary_image, int *labels, int width, int height) {
    int current_label = 1;
    int *equiv = (int *)malloc(MAX_LABELS * sizeof(int));
    
    // Eşdeğerlik tablosunu başlat
    for (int i = 0; i < MAX_LABELS; i++) {
        equiv[i] = i;
    }
    
    // İlk geçiş: etiketleme
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            if (binary_image[idx] == 255) {
                int left_label = (x > 0) ? labels[idx - 1] : 0;
                int top_label = (y > 0) ? labels[(y - 1) * width + x] : 0;
                
                if (left_label == 0 && top_label == 0) {
                    labels[idx] = current_label++;
                } else if (left_label != 0 && top_label == 0) {
                    labels[idx] = left_label;
                } else if (left_label == 0 && top_label != 0) {
                    labels[idx] = top_label;
                } else {
                    // İki etiket var, küçük olanı kullan ve eşdeğerlik kaydet
                    int min_label = (left_label < top_label) ? left_label : top_label;
                    int max_label = (left_label > top_label) ? left_label : top_label;
                    labels[idx] = min_label;
                    equiv[max_label] = min_label;
                }
            } else {
                labels[idx] = BACKGROUND;
            }
        }
    }
    
    // Eşdeğerlik tablosunu düzelt
    for (int i = 1; i < current_label; i++) {
        int root = i;
        while (equiv[root] != root) {
            root = equiv[root];
        }
        equiv[i] = root;
    }
    
    // İkinci geçiş: etiketleri birleştir
    for (int i = 0; i < width * height; i++) {
        if (labels[i] != BACKGROUND) {
            labels[i] = equiv[labels[i]];
        }
    }
    
    free(equiv);
    return current_label - 1;
}

// Blob özelliklerini hesapla
int calculate_blob_properties(int *labels, int width, int height, blob_t *blobs, int max_blobs) {
    int *area = (int *)calloc(MAX_LABELS, sizeof(int));
    int *sum_x = (int *)calloc(MAX_LABELS, sizeof(int));
    int *sum_y = (int *)calloc(MAX_LABELS, sizeof(int));
    int *min_x_arr = (int *)malloc(MAX_LABELS * sizeof(int));
    int *min_y_arr = (int *)malloc(MAX_LABELS * sizeof(int));
    int *max_x_arr = (int *)malloc(MAX_LABELS * sizeof(int));
    int *max_y_arr = (int *)malloc(MAX_LABELS * sizeof(int));
    
    // Min/Max değerleri başlat
    for (int i = 0; i < MAX_LABELS; i++) {
        min_x_arr[i] = width;
        min_y_arr[i] = height;
        max_x_arr[i] = 0;
        max_y_arr[i] = 0;
    }
    
    // Her piksel için blob bilgilerini topla
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int label = labels[idx];
            
            if (label != BACKGROUND) {
                area[label]++;
                sum_x[label] += x;
                sum_y[label] += y;
                
                if (x < min_x_arr[label]) min_x_arr[label] = x;
                if (x > max_x_arr[label]) max_x_arr[label] = x;
                if (y < min_y_arr[label]) min_y_arr[label] = y;
                if (y > max_y_arr[label]) max_y_arr[label] = y;
            }
        }
    }
    
    // Blob yapılarını doldur
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
    
    free(area);
    free(sum_x);
    free(sum_y);
    free(min_x_arr);
    free(min_y_arr);
    free(max_x_arr);
    free(max_y_arr);
    
    return blob_count;
}

// Ana thresholding ve nesne tespit fonksiyonu
int detect_object_by_size(camera_fb_t *fb, blob_t *detected_blob) {
    if (fb == NULL) {
        ESP_LOGE(TAG, "Kamera frame buffer NULL");
        return -1;
    }
    
    int width = fb->width;
    int height = fb->height;
    
    // Bellek tahsisi
    uint8_t *rgb888_image = (uint8_t *)malloc(width * height * 3 * sizeof(uint8_t));
    uint8_t *gray_image = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    uint8_t *binary_image = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    int *labels = (int *)calloc(width * height, sizeof(int));
    
    if (!rgb888_image || !gray_image || !binary_image || !labels) {
        ESP_LOGE(TAG, "Bellek tahsisi hatası");
        free(rgb888_image);
        free(gray_image);
        free(binary_image);
        free(labels);
        return -1;
    }
    
    // RGB565'ten RGB888'e dönüştür
    rgb565_to_rgb888((uint16_t *)fb->buf, rgb888_image, width, height);
    
    // RGB'yi gri tonlamaya çevir
    rgb_to_gray(rgb888_image, gray_image, width, height);
    
    // Otsu threshold hesapla
    int threshold = calculate_otsu_threshold(gray_image, width, height);
    ESP_LOGI(TAG, "Hesaplanan threshold değeri: %d", threshold);
    
    // Thresholding uygula
    apply_threshold(gray_image, binary_image, width, height, threshold);
    
    // Connected components
    int num_labels = label_components(binary_image, labels, width, height);
    ESP_LOGI(TAG, "Tespit edilen bileşen sayısı: %d", num_labels);
    
    // Blob özelliklerini hesapla
    blob_t *blobs = (blob_t *)malloc(MAX_LABELS * sizeof(blob_t));
    int blob_count = calculate_blob_properties(labels, width, height, blobs, MAX_LABELS);
    
    // 1000 piksel boyutuna yakın nesneyi bul
    int found = 0;
    for (int i = 0; i < blob_count; i++) {
        int size_diff = abs(blobs[i].area - TARGET_OBJECT_SIZE);
        if (size_diff <= SIZE_TOLERANCE) {
            *detected_blob = blobs[i];
            found = 1;
            ESP_LOGI(TAG, "Nesne tespit edildi! Boyut: %d piksel, Merkez: (%d, %d)",
                     blobs[i].area, blobs[i].centroid_x, blobs[i].centroid_y);
            break;
        }
    }
    
    // Bellek temizle
    free(rgb888_image);
    free(gray_image);
    free(binary_image);
    free(labels);
    free(blobs);
    
    return found ? 0 : -1;
}

// ESP32 CAM başlatma ve thresholding örneği
void app_main(void) {
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
    config.pixel_format = PIXFORMAT_RGB565; // RGB565 formatı
    config.frame_size = FRAMESIZE_VGA; // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 1;
    
    // Kamera başlat
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Kamera başlatma hatası: %s", esp_err_to_name(err));
        return;
    }
    
    ESP_LOGI(TAG, "Kamera başlatıldı");
    
    // Ana döngü
    while (1) {
        // Frame yakala
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Frame yakalama hatası");
            vTaskDelay(1000 / portTICK_PERIOD_MS);
            continue;
        }
        
        blob_t detected_blob;
        int result = detect_object_by_size(fb, &detected_blob);
        
        if (result == 0) {
            ESP_LOGI(TAG, "Nesne tespit edildi!");
            ESP_LOGI(TAG, "  Boyut: %d piksel", detected_blob.area);
            ESP_LOGI(TAG, "  Merkez: (%d, %d)", detected_blob.centroid_x, detected_blob.centroid_y);
            ESP_LOGI(TAG, "  Bounding Box: (%d, %d) - (%d, %d)",
                     detected_blob.min_x, detected_blob.min_y,
                     detected_blob.max_x, detected_blob.max_y);
        } else {
            ESP_LOGI(TAG, "1000 piksel boyutunda nesne tespit edilemedi");
        }
        
        // Frame'i serbest bırak
        esp_camera_fb_return(fb);
        
        // 1 saniye bekle
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}
