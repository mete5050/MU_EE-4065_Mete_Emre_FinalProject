import os
import json
import numpy as np
import tensorflow as tf

# ============================================================
# Config
# ============================================================
OUT_DIR = "q4_out"
os.makedirs(OUT_DIR, exist_ok=True)

IMG = 32
CH = 3
NUM_CLASSES = 10

EPOCHS = 5            # demo için 5-10 yeter; daha iyi için 20+
BATCH = 128
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# Data (MNIST -> 32x32 -> 3ch)
# ============================================================
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # [N,28,28] -> [N,28,28,1]
    x_train = x_train[..., None].astype(np.float32)
    x_test  = x_test[..., None].astype(np.float32)

    # resize to 32x32
    x_train = tf.image.resize(x_train, (IMG, IMG)).numpy()
    x_test  = tf.image.resize(x_test,  (IMG, IMG)).numpy()

    # replicate to 3 channels
    x_train = np.repeat(x_train, CH, axis=-1)
    x_test  = np.repeat(x_test,  CH, axis=-1)

    # normalize [0,1]
    x_train /= 255.0
    x_test  /= 255.0

    # one-hot
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

    return (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh)

# Representative dataset for int8 calibration
def representative_dataset(x_train, n=200):
    idx = np.random.choice(len(x_train), size=n, replace=False)
    for i in idx:
        # converter expects float32 input sample
        yield [x_train[i:i+1].astype(np.float32)]

# ============================================================
# Models (ESP32-friendly small CNNs)
# ============================================================
def model_squeezenet_mini():
    inp = tf.keras.Input((IMG, IMG, CH))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D()(x)

    # "fire" blocks (squeeze + expand)
    def fire(x, s, e):
        sq = tf.keras.layers.Conv2D(s, 1, activation="relu", padding="same")(x)
        e1 = tf.keras.layers.Conv2D(e, 1, activation="relu", padding="same")(sq)
        e3 = tf.keras.layers.Conv2D(e, 3, activation="relu", padding="same")(sq)
        return tf.keras.layers.Concatenate()([e1, e3])

    x = fire(x, 8, 16)
    x = tf.keras.layers.MaxPool2D()(x)
    x = fire(x, 12, 24)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="squeezenetmini")

def model_mobilenetv2_mini():
    inp = tf.keras.Input((IMG, IMG, CH))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # depthwise separable blocks
    def dws(x, pw, stride=1):
        x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        x = tf.keras.layers.Conv2D(pw, 1, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        return x

    x = dws(x, 24, stride=2)
    x = dws(x, 32, stride=2)
    x = dws(x, 48, stride=1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="mobilenetv2mini")

def model_resnet8():
    inp = tf.keras.Input((IMG, IMG, CH))

    def res_block(x, f, stride=1):
        shortcut = x
        x = tf.keras.layers.Conv2D(f, 3, strides=stride, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != f:
            shortcut = tf.keras.layers.Conv2D(f, 1, strides=stride, padding="same", use_bias=False)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = res_block(x, 16, 1)
    x = res_block(x, 24, 2)
    x = res_block(x, 32, 2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="resnet8")

def model_efficientnet_mini():
    # EfficientNet-lite gibi davranan küçük MBConv blokları
    inp = tf.keras.Input((IMG, IMG, CH))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)

    def mbconv(x, out_ch, expand=2, stride=1):
        in_ch = x.shape[-1]
        # expand
        y = tf.keras.layers.Conv2D(int(in_ch*expand), 1, padding="same", use_bias=False)(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation("swish")(y)
        # depthwise
        y = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation("swish")(y)
        # squeeze-excite (tiny)
        se = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(y)
        se = tf.keras.layers.Conv2D(max(4, int(out_ch//4)), 1, activation="swish")(se)
        se = tf.keras.layers.Conv2D(y.shape[-1], 1, activation="sigmoid")(se)
        y = tf.keras.layers.Multiply()([y, se])
        # project
        y = tf.keras.layers.Conv2D(out_ch, 1, padding="same", use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)

        if stride == 1 and in_ch == out_ch:
            y = tf.keras.layers.Add()([x, y])
        return y

    x = mbconv(x, 24, expand=2, stride=2)
    x = mbconv(x, 24, expand=2, stride=1)
    x = mbconv(x, 32, expand=2, stride=2)
    x = mbconv(x, 32, expand=2, stride=1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="efficientnetmini")

# ============================================================
# Train / Evaluate
# ============================================================
def train_model(model, x_train, y_train_oh, x_test, y_test_oh):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train, y_train_oh,
        validation_data=(x_test, y_test_oh),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=2
    )
    return model

# ============================================================
# TFLite int8 export + scales/zp extract
# ============================================================
def export_int8_tflite(model, x_train, tflite_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(x_train, n=200)

    # full int8 (ESP32 TFLite Micro)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

def get_quant_params(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]
    return {
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zp),
        "output_scale": float(out_scale),
        "output_zero_point": int(out_zp),
        "input_shape": list(in_det["shape"]),
        "output_shape": list(out_det["shape"]),
        "input_dtype": str(in_det["dtype"]),
        "output_dtype": str(out_det["dtype"]),
    }

def bin_to_c_array(bin_path, var_name):
    data = open(bin_path, "rb").read()
    # C array string (PROGMEM friendly)
    hex_bytes = ",".join(f"0x{b:02x}" for b in data)
    return (
        f"// Auto-generated from {os.path.basename(bin_path)}\n"
        f"const unsigned char {var_name}[] PROGMEM = {{{hex_bytes}}};\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
    )

def write_model_data_h(models_meta, out_h_path):
    with open(out_h_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <Arduino.h>\n\n")

        for m in models_meta:
            f.write(bin_to_c_array(m["tflite_path"], m["c_name"]))
            f.write("\n")
            f.write(f"// Quant params for {m['c_name']}\n")
            f.write(f"const float {m['c_name']}_input_scale = {m['q']['input_scale']:.12f}f;\n")
            f.write(f"const int {m['c_name']}_input_zero_point = {m['q']['input_zero_point']};\n")
            f.write(f"const float {m['c_name']}_output_scale = {m['q']['output_scale']:.12f}f;\n")
            f.write(f"const int {m['c_name']}_output_zero_point = {m['q']['output_zero_point']};\n\n")

# ============================================================
# Main
# ============================================================
def main():
    (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh) = load_data()

    models = [
        ("squeezenetmini", model_squeezenet_mini(), "squeezenetmini_model"),
        ("mobilenetv2mini", model_mobilenetv2_mini(), "mobilenetv2mini_model"),
        ("resnet8", model_resnet8(), "resnet8_model"),
        ("efficientnetmini", model_efficientnet_mini(), "efficientnetmini_model"),
    ]

    meta = []

    for name, mdl, c_name in models:
        print("\n==============================")
        print("Training:", name)
        print("==============================")
        mdl = train_model(mdl, x_train, y_train_oh, x_test, y_test_oh)

        # Save keras
        keras_path = os.path.join(OUT_DIR, f"{name}.keras")
        mdl.save(keras_path)

        # Export int8 tflite
        tflite_path = os.path.join(OUT_DIR, f"{name}_int8.tflite")
        export_int8_tflite(mdl, x_train, tflite_path)

        q = get_quant_params(tflite_path)
        print(f"[{name}] quant:", q)

        meta.append({
            "name": name,
            "c_name": c_name,
            "tflite_path": tflite_path,
            "q": q
        })

    # Write metadata json
    with open(os.path.join(OUT_DIR, "models_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Write model_data.h
    out_h = os.path.join(OUT_DIR, "model_data.h")
    write_model_data_h(meta, out_h)
    print("\nDONE.")
    print("Generated:", out_h)
    print("Copy model_data.h into Arduino sketch folder.")

if __name__ == "__main__":
    main()
