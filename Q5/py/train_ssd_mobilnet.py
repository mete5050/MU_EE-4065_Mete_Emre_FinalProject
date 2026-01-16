"""
Soru 5b: SSD+MobileNet ile el yazısı rakam tespiti
Single Shot Detector (SSD) + MobileNet backbone
"""

import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
import json

# ESP32 CAM için optimizasyon
ESP32_INPUT_SIZE = 160  # QQVGA benzeri (RAM kısıtları için)
ESP32_NUM_CLASSES = 10
NUM_ANCHORS = 1  # Her grid cell için anchor sayısı (ESP32 için agresif optimizasyon - 3'ten 1'e düşürüldü)
ESP32_GRID_SIZE = 16  # 160/10 = 16 (daha küçük grid, daha az output tensor)

def create_ssd_mobilenet(input_shape=(ESP32_INPUT_SIZE, ESP32_INPUT_SIZE, 1), num_classes=ESP32_NUM_CLASSES):
    """
    SSD + MobileNet modeli oluştur
    
    SSD (Single Shot Detector):
    - Tek aşamalı object detection
    - MobileNet backbone ile hızlı inference
    - Anchor-based detection
    """
    # MobileNetV2 backbone (ESP32 RAM kısıtlarına göre optimize edilmiş)
    # NOT: ImageNet weights ile sadece 0.35, 0.50, 0.75, 1.0, 1.3, 1.4 kullanılabilir
    base_model = keras.applications.MobileNetV2(
        input_shape=(ESP32_INPUT_SIZE, ESP32_INPUT_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35  # En küçük geçerli değer (ESP32 için optimize)
    )
    
    # Input layer (grayscale -> RGB)
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(3, (1, 1), padding='same', name='grayscale_to_rgb')(inputs)
    
    # Backbone
    backbone_output = base_model(x, training=False)
    
    # Backbone output boyutunu kontrol et ve upsampling yap
    # MobileNetV2 160x160 input için 5x5 feature map üretir (32x downsampling)
    # ESP32 için 16x16 grid istiyoruz (daha küçük output), bu yüzden ~3.2x upsampling yapıyoruz
    
    # Upsampling: 5x5 -> 16x16 (3.2x upsampling, en yakın 3x kullanılıyor)
    ssd_layer1 = layers.UpSampling2D(size=(3, 3), interpolation='bilinear')(backbone_output)  # 5*3 = 15, padding ile 16'ya tamamlanır
    # 15x15'ten 16x16'ya tamamlamak için padding
    ssd_layer1 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(ssd_layer1)  # 15x15 -> 16x16
    
    # SSD detection layers (ESP32 RAM agresif optimizasyon - model boyutunu 1/4'e düşürmek için)
    # Channel sayısını agresif azalt: 128 -> 32 (model boyutunu ~75% azaltır)
    ssd_layer1 = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(ssd_layer1)
    ssd_layer1 = layers.BatchNormalization()(ssd_layer1)
    ssd_layer1 = layers.ReLU()(ssd_layer1)
    ssd_layer1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu', use_bias=False)(ssd_layer1)  # 128 -> 32 (agresif optimizasyon)
    ssd_layer1 = layers.BatchNormalization()(ssd_layer1)
    
    # Detection heads
    # Grid size: 16x16 (ESP32 için agresif optimizasyon - model boyutunu 1/4'e düşürmek için)
    grid_size_1 = ESP32_GRID_SIZE  # 16
    
    # Detection head 1 (16x16) - ESP32 için tek detection head, agresif optimizasyon
    # Kernel size'ı küçült: 3x3 -> 1x1 (daha az parametre, ~9x azalma)
    det1_loc = layers.Conv2D(NUM_ANCHORS * 4, (1, 1), padding='same', use_bias=False, name='det1_loc')(ssd_layer1)  # 3x3 -> 1x1
    det1_conf = layers.Conv2D(NUM_ANCHORS * (num_classes + 1), (1, 1), padding='same', use_bias=False, name='det1_conf')(ssd_layer1)  # 3x3 -> 1x1
    
    # Reshape outputs (16x16 grid)
    det1_loc = layers.Reshape((grid_size_1, grid_size_1, NUM_ANCHORS, 4))(det1_loc)
    det1_conf = layers.Reshape((grid_size_1, grid_size_1, NUM_ANCHORS, num_classes + 1))(det1_conf)
    det1_conf = layers.Activation('softmax', name='det1_conf_softmax')(det1_conf)
    
    # ESP32 için sadece tek detection head kullanılıyor (RAM kısıtları)
    outputs = {
        'loc': det1_loc,
        'conf': det1_conf
    }
    
    model = models.Model(inputs, outputs, name='ssd_mobilenet')
    return model

def ssd_loss(num_classes, num_anchors):
    """
    SSD loss fonksiyonu (localization + confidence)
    """
    def loss_fn(y_true, y_pred):
        # y_true: [batch, grid_h, grid_w, anchors, 5] (x, y, w, h, class)
        # y_pred: {'loc': [batch, grid_h, grid_w, anchors, 4], 
        #          'conf': [batch, grid_h, grid_w, anchors, num_classes+1]}
        
        # Basitleştirilmiş loss
        # Gerçek implementasyonda smooth L1 loss ve focal loss kullanılır
        loc_pred = y_pred['loc']
        conf_pred = y_pred['conf']
        
        # Placeholder - gerçek loss implementasyonu gerekli
        return tf.constant(0.0)
    
    return loss_fn

def prepare_ssd_dataset(dataset_path):
    """
    SSD için dataset hazırlama
    """
    train_dir = os.path.join(dataset_path, "train", "images")
    labels_dir = os.path.join(dataset_path, "train", "labels")
    
    grid_size = ESP32_GRID_SIZE  # 20 (ESP32 için optimize edilmiş)
    
    def yolo_to_ssd_format(label_path, grid_size, num_classes, num_anchors):
        """
        YOLO label'ını SSD formatına dönüştür
        SSD için iki ayrı output: loc ve conf
        """
        # SSD loc output: [grid_h, grid_w, anchors, 4] (x, y, w, h)
        ssd_loc = np.zeros((grid_size, grid_size, num_anchors, 4), dtype=np.float32)
        # SSD conf output: [grid_h, grid_w, anchors, num_classes + 1] (one-hot encoded)
        ssd_conf = np.zeros((grid_size, grid_size, num_anchors, num_classes + 1), dtype=np.float32)
        
        if not os.path.exists(label_path):
            # Background için tüm conf'u background class'a ayarla
            ssd_conf[:, :, :, num_classes] = 1.0
            return ssd_loc, ssd_conf
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Grid koordinatları
            grid_x = int(center_x * grid_size)
            grid_y = int(center_y * grid_size)
            
            grid_x = max(0, min(grid_x, grid_size - 1))
            grid_y = max(0, min(grid_y, grid_size - 1))
            
            # İlk anchor'a yaz (basitleştirilmiş)
            # Location (bounding box)
            ssd_loc[grid_y, grid_x, 0, 0] = center_x
            ssd_loc[grid_y, grid_x, 0, 1] = center_y
            ssd_loc[grid_y, grid_x, 0, 2] = width
            ssd_loc[grid_y, grid_x, 0, 3] = height
            
            # Confidence (one-hot encoded: class_id = 1, background = 0)
            ssd_conf[grid_y, grid_x, 0, class_id] = 1.0
            ssd_conf[grid_y, grid_x, 0, num_classes] = 0.0  # Not background
        
        # Background için normalize et (eğer hiç object yoksa)
        for y in range(grid_size):
            for x in range(grid_size):
                for a in range(num_anchors):
                    if np.sum(ssd_conf[y, x, a, :num_classes]) == 0:
                        ssd_conf[y, x, a, num_classes] = 1.0  # Background
        
        return ssd_loc, ssd_conf
    
    class SSDDatasetGenerator(keras.utils.Sequence):
        def __init__(self, image_dir, label_dir, batch_size=16, shuffle=True):
            self.image_dir = image_dir
            self.label_dir = label_dir
            self.batch_size = batch_size
            self.shuffle = shuffle
            
            self.image_files = [f for f in os.listdir(image_dir) 
                               if f.endswith(('.jpg', '.png'))]
            self.indices = np.arange(len(self.image_files))
            
            if shuffle:
                np.random.shuffle(self.indices)
        
        def __len__(self):
            return len(self.image_files) // self.batch_size
        
        def __getitem__(self, idx):
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_images = []
            batch_labels_loc = []
            batch_labels_conf = []
            
            for i in batch_indices:
                img_file = self.image_files[i]
                img_path = os.path.join(self.image_dir, img_file)
                label_path = os.path.join(self.label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                
                # Görüntü yükle
                img = keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(ESP32_INPUT_SIZE, ESP32_INPUT_SIZE))
                img = keras.preprocessing.image.img_to_array(img) / 255.0
                batch_images.append(img)
                
                # SSD formatına dönüştür (loc ve conf ayrı)
                ssd_loc, ssd_conf = yolo_to_ssd_format(label_path, grid_size, ESP32_NUM_CLASSES, NUM_ANCHORS)
                batch_labels_loc.append(ssd_loc)
                batch_labels_conf.append(ssd_conf)
            
            return np.array(batch_images), {
                'loc': np.array(batch_labels_loc),
                'conf': np.array(batch_labels_conf)
            }
        
        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)
    
    train_gen = SSDDatasetGenerator(
        os.path.join(dataset_path, "train", "images"),
        os.path.join(dataset_path, "train", "labels"),
        batch_size=16
    )
    
    val_gen = SSDDatasetGenerator(
        os.path.join(dataset_path, "val", "images"),
        os.path.join(dataset_path, "val", "labels"),
        batch_size=16,
        shuffle=False
    )
    
    return train_gen, val_gen

def train_ssd_mobilenet(dataset_path, epochs=50):
    """
    SSD+MobileNet modeli eğit
    """
    print("SSD+MobileNet Model Eğitimi Başlatılıyor...")
    
    # Dataset hazırla
    train_gen, val_gen = prepare_ssd_dataset(dataset_path)
    
    # Model oluştur
    model = create_ssd_mobilenet()
    
    # Loss ve optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={'loc': 'mse', 'conf': 'categorical_crossentropy'},
        loss_weights={'loc': 1.0, 'conf': 1.0},
        metrics={'conf': 'accuracy'}
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/ssd_mobilenet_best.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Eğitim
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\nSSD+MobileNet Model Özeti:")
    print(f"Parametre sayısı: {model.count_params():,}")
    model.summary()
    
    return model, history

def convert_ssd_to_tflite(model_path, output_path, quantize=True, dataset_path=None):
    """
    SSD+MobileNet modelini TFLite formatına dönüştür (ESP32 için optimize edilmiş)
    - INT8 quantization zorunlu (RAM kısıtları için)
    - Model boyutu < 400KB hedefleniyor
    """
    print(f"SSD+MobileNet modeli TFLite'a dönüştürülüyor: {model_path}")
    # compile=False: TFLite'a dönüştürürken optimizer ve loss'a ihtiyaç yok
    # Keras 3.x'te custom loss fonksiyonları sorun çıkarabilir, bu yüzden compile=False kullanıyoruz
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        # Eğer hala sorun varsa, model'i weights olarak yükle
        print(f"UYARI: Model yüklenirken hata: {e}")
        print("Alternatif yöntem deneniyor: Model architecture + weights...")
        # Model'i yeniden oluştur ve weights yükle
        weights_path = model_path.replace('_best.h5', '_weights.h5').replace('.h5', '_weights.h5')
        if os.path.exists(weights_path):
            model = create_ssd_mobilenet()
            model.load_weights(weights_path)
        else:
            raise Exception(f"Model yüklenemedi ve weights dosyası bulunamadı: {weights_path}")
    
    # Model boyutu kontrolü
    param_count = model.count_params()
    print(f"Model parametre sayısı: {param_count:,}")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("Quantization uygulanıyor (ESP32 için optimize)...")
        
        # Önce dynamic range quantization dene (daha uyumlu)
        try:
            print("Dynamic range quantization deneniyor...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Representative dataset (gerçek veri kullanılırsa daha iyi quantization)
            def representative_dataset():
                if dataset_path and os.path.exists(dataset_path):
                    # Gerçek dataset'ten örnekler kullan
                    train_dir = os.path.join(dataset_path, "train", "images")
                    if os.path.exists(train_dir):
                        image_files = [f for f in os.listdir(train_dir) 
                                     if f.endswith(('.jpg', '.png'))][:100]
                        for img_file in image_files:
                            img_path = os.path.join(train_dir, img_file)
                            try:
                                img = keras.preprocessing.image.load_img(
                                    img_path, color_mode='grayscale', 
                                    target_size=(ESP32_INPUT_SIZE, ESP32_INPUT_SIZE)
                                )
                                img_array = keras.preprocessing.image.img_to_array(img)
                                img_array = img_array / 255.0  # Normalize to [0, 1]
                                img_array = tf.expand_dims(img_array, 0)
                                yield [img_array]
                            except Exception as e:
                                continue
                # Fallback: Random normalized data
                for _ in range(100):
                    yield [np.random.rand(1, ESP32_INPUT_SIZE, ESP32_INPUT_SIZE, 1).astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            
            # Önce dynamic range quantization ile dene
            print("Dynamic range quantization ile dönüştürülüyor...")
            tflite_model = converter.convert()
            
            # Başarılı olursa, INT8 quantization'ı ayrı bir dosya olarak dene
            print("INT8 quantization deneniyor (daha küçük model)...")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            # Representative dataset'i uint8 formatına çevir
            def representative_dataset_int8():
                if dataset_path and os.path.exists(dataset_path):
                    train_dir = os.path.join(dataset_path, "train", "images")
                    if os.path.exists(train_dir):
                        image_files = [f for f in os.listdir(train_dir) 
                                     if f.endswith(('.jpg', '.png'))][:100]
                        for img_file in image_files:
                            img_path = os.path.join(train_dir, img_file)
                            try:
                                img = keras.preprocessing.image.load_img(
                                    img_path, color_mode='grayscale', 
                                    target_size=(ESP32_INPUT_SIZE, ESP32_INPUT_SIZE)
                                )
                                img_array = keras.preprocessing.image.img_to_array(img)
                                img_array = tf.expand_dims(img_array, 0)
                                img_array = tf.cast(img_array, tf.uint8)
                                yield [img_array]
                            except Exception as e:
                                continue
                for _ in range(100):
                    yield [np.random.randint(0, 256, (1, ESP32_INPUT_SIZE, ESP32_INPUT_SIZE, 1), dtype=np.uint8)]
            
            converter.representative_dataset = representative_dataset_int8
            
            try:
                tflite_model_int8 = converter.convert()
                # INT8 başarılı olursa onu kullan
                tflite_model = tflite_model_int8
                print("✓ INT8 quantization başarılı!")
            except Exception as e:
                print(f"UYARI: INT8 quantization başarısız, dynamic range quantization kullanılıyor: {e}")
                # Dynamic range quantization sonucunu kullan
                
        except Exception as e:
            print(f"UYARI: Quantization başarısız, float model kullanılıyor: {e}")
            quantize = False
            tflite_model = converter.convert()
    else:
        # Quantization yok, direkt dönüştür
        tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = os.path.getsize(output_path) / 1024
    size_mb = size_kb / 1024
    print(f"SSD+MobileNet TFLite model kaydedildi: {output_path}")
    print(f"Model boyutu: {size_kb:.2f} KB ({size_mb:.2f} MB)")
    
    if size_kb > 400:
        print(f"UYARI: Model boyutu {size_kb:.2f} KB, ESP32 için ideal < 400KB")
    else:
        print(f"✓ Model boyutu ESP32 için uygun: {size_kb:.2f} KB")

if __name__ == "__main__":
    # Script'in bulunduğu klasörde dataset'i ara
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "handwritten_digits_dataset")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset bulunamadı: {dataset_path}")
        print("Önce 'python prepare_dataset_from_raw.py' çalıştırarak dataset hazırlayın.")
        exit(1)
    
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # SSD+MobileNet modeli eğit
    model, history = train_ssd_mobilenet(dataset_path, epochs=30)
    
    # TFLite'a dönüştür (ESP32 için optimize edilmiş)
    convert_ssd_to_tflite(
        os.path.join(models_dir, "ssd_mobilenet_best.h5"), 
        os.path.join(models_dir, "ssd_mobilenet_esp32.tflite"), 
        quantize=True,
        dataset_path=dataset_path
    )
    
    print("\nSSD+MobileNet eğitimi tamamlandı!")
    print(f"Model '{os.path.join(models_dir, 'ssd_mobilenet_esp32.tflite')}' olarak kaydedildi")
