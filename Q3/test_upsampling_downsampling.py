"""
Soru 3: PC üzerinde Upsampling ve Downsampling Test Kodu

Assignment Requirements:
- a- (10 points) perform upsampling with a given value
- b- (10 points) perform downsampling with a given value
- Support for non-integer scale factors (1.5, 2/3, etc.)
"""

import cv2
import numpy as np
import argparse
import os
import sys


# ============================================================================
# INTERPOLATION FUNCTIONS
# ============================================================================

def bilinear_interpolate(p00, p01, p10, p11, fx, fy):
    """Çift doğrusal interpolasyon"""
    a = p00 * (1.0 - fx) * (1.0 - fy)
    b = p01 * fx * (1.0 - fy)
    c = p10 * (1.0 - fx) * fy
    d = p11 * fx * fy
    return int(a + b + c + d + 0.5)


def cubic_weight(x):
    """Bicubic interpolation için ağırlık fonksiyonu"""
    abs_x = abs(x)
    if abs_x <= 1.0:
        return 1.5 * abs_x**3 - 2.5 * abs_x**2 + 1.0
    elif abs_x <= 2.0:
        return -0.5 * abs_x**3 + 2.5 * abs_x**2 - 4.0 * abs_x + 2.0
    return 0.0


def bicubic_interpolate(pixels, fx, fy):
    """Çift kübik interpolasyon"""
    result = 0.0
    for j in range(4):
        for i in range(4):
            weight = cubic_weight(i - 1.0 - fx) * cubic_weight(j - 1.0 - fy)
            result += pixels[j][i] * weight
    return int(max(0, min(255, result + 0.5)))


# ============================================================================
# UPSAMPLING AND DOWNSAMPLING FUNCTIONS
# ============================================================================

def upsampling(image, scale_factor, method='bilinear'):
    """
    Upsampling - Görüntüyü büyütme
    
    Args:
        image: Giriş görüntü (numpy array, grayscale veya color)
        scale_factor: Ölçek faktörü (örn: 1.5, 2.0)
        method: 'nearest', 'bilinear', 'bicubic'
    
    Returns:
        Upsampled görüntü
    """
    h, w = image.shape[:2]
    new_w = int(w * scale_factor + 0.5)
    new_h = int(h * scale_factor + 0.5)
    
    scale_x = w / new_w
    scale_y = h / new_h
    
    if len(image.shape) == 2:
        output = np.zeros((new_h, new_w), dtype=np.uint8)
        
        if method == 'nearest':
            for y in range(new_h):
                for x in range(new_w):
                    src_x = int(x * scale_x + 0.5)
                    src_y = int(y * scale_y + 0.5)
                    src_x = min(src_x, w - 1)
                    src_y = min(src_y, h - 1)
                    output[y, x] = image[src_y, src_x]
        
        elif method == 'bilinear':
            for y in range(new_h):
                for x in range(new_w):
                    src_x_f = x * scale_x
                    src_y_f = y * scale_y
                    src_x = int(src_x_f)
                    src_y = int(src_y_f)
                    fx = src_x_f - src_x
                    fy = src_y_f - src_y
                    
                    x0 = min(src_x, w - 1)
                    y0 = min(src_y, h - 1)
                    x1 = min(src_x + 1, w - 1)
                    y1 = min(src_y + 1, h - 1)
                    
                    p00 = image[y0, x0]
                    p01 = image[y0, x1]
                    p10 = image[y1, x0]
                    p11 = image[y1, x1]
                    
                    output[y, x] = bilinear_interpolate(p00, p01, p10, p11, fx, fy)
        
        elif method == 'bicubic':
            for y in range(new_h):
                for x in range(new_w):
                    src_x_f = x * scale_x
                    src_y_f = y * scale_y
                    src_x = int(src_x_f)
                    src_y = int(src_y_f)
                    fx = src_x_f - src_x
                    fy = src_y_f - src_y
                    
                    pixels = np.zeros((4, 4), dtype=np.uint8)
                    for j in range(4):
                        for i in range(4):
                            px = src_x + i - 1
                            py = src_y + j - 1
                            px = max(0, min(px, w - 1))
                            py = max(0, min(py, h - 1))
                            pixels[j, i] = image[py, px]
                    
                    output[y, x] = bicubic_interpolate(pixels, fx, fy)
    else:
        output = np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8)
        for c in range(image.shape[2]):
            output[:, :, c] = upsampling(image[:, :, c], scale_factor, method)
    
    return output


def downsampling(image, scale_factor, method='bilinear'):
    """
    Downsampling - Görüntüyü küçültme
    
    Args:
        image: Giriş görüntü (numpy array, grayscale veya color)
        scale_factor: Ölçek faktörü (örn: 0.5, 0.6)
        method: 'nearest', 'bilinear', 'bicubic'
    
    Returns:
        Downsampled görüntü
    """
    h, w = image.shape[:2]
    new_w = max(1, int(w * scale_factor + 0.5))
    new_h = max(1, int(h * scale_factor + 0.5))
    
    scale_x = w / new_w
    scale_y = h / new_h
    
    if len(image.shape) == 2:
        output = np.zeros((new_h, new_w), dtype=np.uint8)
        
        if method == 'nearest':
            for y in range(new_h):
                for x in range(new_w):
                    src_x = int(x * scale_x + 0.5)
                    src_y = int(y * scale_y + 0.5)
                    src_x = min(src_x, w - 1)
                    src_y = min(src_y, h - 1)
                    output[y, x] = image[src_y, src_x]
        
        elif method == 'bilinear':
            for y in range(new_h):
                for x in range(new_w):
                    src_x_f = x * scale_x
                    src_y_f = y * scale_y
                    src_x = int(src_x_f)
                    src_y = int(src_y_f)
                    fx = src_x_f - src_x
                    fy = src_y_f - src_y
                    
                    x0 = min(src_x, w - 1)
                    y0 = min(src_y, h - 1)
                    x1 = min(src_x + 1, w - 1)
                    y1 = min(src_y + 1, h - 1)
                    
                    p00 = image[y0, x0]
                    p01 = image[y0, x1]
                    p10 = image[y1, x0]
                    p11 = image[y1, x1]
                    
                    output[y, x] = bilinear_interpolate(p00, p01, p10, p11, fx, fy)
        
        elif method == 'bicubic':
            for y in range(new_h):
                for x in range(new_w):
                    src_x_f = x * scale_x
                    src_y_f = y * scale_y
                    src_x = int(src_x_f)
                    src_y = int(src_y_f)
                    fx = src_x_f - src_x
                    fy = src_y_f - src_y
                    
                    pixels = np.zeros((4, 4), dtype=np.uint8)
                    for j in range(4):
                        for i in range(4):
                            px = src_x + i - 1
                            py = src_y + j - 1
                            px = max(0, min(px, w - 1))
                            py = max(0, min(py, h - 1))
                            pixels[j, i] = image[py, px]
                    
                    output[y, x] = bicubic_interpolate(pixels, fx, fy)
    else:
        output = np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8)
        for c in range(image.shape[2]):
            output[:, :, c] = downsampling(image[:, :, c], scale_factor, method)
    
    return output


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_image(image_path):
    """Resim dosyasını yükle (Türkçe karakter sorununu çözer)"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"Resim yüklenemedi: {image_path}")
    
    return image


def save_image(image, output_path):
    """Resmi kaydet (Türkçe karakter sorununu çözer)"""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.jpg' or ext == '.jpeg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif ext == '.png':
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    else:
        encode_param = []
    
    success, encoded_img = cv2.imencode(ext, image, encode_param)
    if not success:
        raise IOError(f"Resim encode edilemedi: {output_path}")
    
    with open(output_path, 'wb') as f:
        f.write(encoded_img.tobytes())


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Upsampling ve Downsampling Test')
    parser.add_argument('--image', type=str, default='input_image.jpg',
                       help='İşlenecek resim dosyası (varsayılan: input_image.jpg)')
    parser.add_argument('--upscale', type=float, default=1.5,
                       help='Upsampling ölçek faktörü (varsayılan: 1.5)')
    parser.add_argument('--downscale', type=float, default=0.6,
                       help='Downsampling ölçek faktörü (varsayılan: 0.6)')
    parser.add_argument('--method', type=str, default='bilinear',
                       choices=['nearest', 'bilinear', 'bicubic'],
                       help='Interpolasyon yöntemi')
    
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\n{'='*70}")
    print(" " * 15 + "UPSAMPLING VE DOWNSAMPLING TEST")
    print(f"{'='*70}\n")
    
    # Dosya yolunu belirle
    if os.path.isabs(args.image):
        image_path = args.image
    else:
        image_path = os.path.join(script_dir, args.image)
    
    print(f"Resim yolu: {image_path}")
    
    # Resmi yükle
    try:
        file_size = os.path.getsize(image_path)
        print(f"[OK] Dosya bulundu ({file_size:,} bytes)")
        
        image = load_image(image_path)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        original_h, original_w = gray.shape
        print(f"[OK] Resim yüklendi: {original_w}x{original_h} piksel\n")
        
    except Exception as e:
        print(f"[HATA] {e}")
        print(f"\nLUTFEN SUNLARI KONTROL EDIN:")
        print(f"   1. Resim dosyasi su klasorde olmali: {script_dir}")
        print("   2. Dosya adi dogru yazilmis olmali (orn: input_image.jpg)")
        print("   3. Desteklenen formatlar: .jpg, .jpeg, .png, .bmp")
        sys.exit(1)
    
    # Upsampling
    print(f"{'─'*70}")
    print(f"UPSAMPLING ({args.upscale}x) - {args.method.upper()}")
    print(f"{'─'*70}")
    print(f"  Orijinal boyut: {original_w}x{original_h} piksel")
    
    try:
        upsampled = upsampling(gray, args.upscale, method=args.method)
        upsampled_h, upsampled_w = upsampled.shape
        print(f"  Yeni boyut:    {upsampled_w}x{upsampled_h} piksel")
        
        upsampled_filename = os.path.join(script_dir, 
                                         f"upsampled_{args.upscale}x_{args.method}.jpg")
        save_image(upsampled, upsampled_filename)
        
        if os.path.exists(upsampled_filename):
            file_size = os.path.getsize(upsampled_filename)
            print(f"  [OK] Kaydedildi: {os.path.basename(upsampled_filename)} ({file_size:,} bytes)\n")
        else:
            print(f"  [HATA] Dosya kaydedilemedi!\n")
    except Exception as e:
        print(f"  [HATA] {e}\n")
        import traceback
        traceback.print_exc()
    
    # Downsampling
    print(f"{'─'*70}")
    print(f"DOWNSAMPLING ({args.downscale}x) - {args.method.upper()}")
    print(f"{'─'*70}")
    print(f"  Orijinal boyut: {original_w}x{original_h} piksel")
    
    try:
        downsampled = downsampling(gray, args.downscale, method=args.method)
        downsampled_h, downsampled_w = downsampled.shape
        print(f"  Yeni boyut:    {downsampled_w}x{downsampled_h} piksel")
        
        downsampled_filename = os.path.join(script_dir,
                                           f"downsampled_{args.downscale}x_{args.method}.jpg")
        save_image(downsampled, downsampled_filename)
        
        if os.path.exists(downsampled_filename):
            file_size = os.path.getsize(downsampled_filename)
            print(f"  [OK] Kaydedildi: {os.path.basename(downsampled_filename)} ({file_size:,} bytes)\n")
        else:
            print(f"  [HATA] Dosya kaydedilemedi!\n")
    except Exception as e:
        print(f"  [HATA] {e}\n")
        import traceback
        traceback.print_exc()
    
    # Özet
    print(f"{'='*70}")
    print(" " * 25 + "[OK] ISLEM TAMAMLANDI!")
    print(f"{'='*70}")
    print(f"  Calisma klasoru: {script_dir}")
    print(f"  Orijinal resim:   {os.path.basename(image_path)}")
    print(f"  Upsampled resim:  upsampled_{args.upscale}x_{args.method}.jpg")
    print(f"  Downsampled resim: downsampled_{args.downscale}x_{args.method}.jpg")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
