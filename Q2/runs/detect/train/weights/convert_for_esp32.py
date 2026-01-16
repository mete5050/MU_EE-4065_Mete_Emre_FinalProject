import argparse
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("STDOUT:\n", p.stdout)
        print("STDERR:\n", p.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if p.stdout.strip():
        print(p.stdout)

def find_first_tflite(out_dir: Path) -> Path | None:
    # onnx2tf bazen direkt tflite üretir, bazen saved_model üretir
    for p in out_dir.rglob("*.tflite"):
        return p
    return None

def load_rep_images(rep_dir: Path, n: int, input_hw: tuple[int,int], channels: int) -> np.ndarray:
    """Return uint8 batch (N,H,W,C)"""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [p for p in rep_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"Representative dataset folder has no images: {rep_dir}")

    files = sorted(files)[:n]
    H, W = input_hw
    batch = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img = img.resize((W, H), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # (H,W,3)
        if channels == 1:
            # RGB -> grayscale
            arr = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8)
            arr = arr[:, :, None]  # (H,W,1)
        batch.append(arr)
    batch = np.stack(batch, axis=0)  # (N,H,W,C)
    return batch

def write_c_array(tflite_path: Path, out_h: Path, var_name: str = "g_model") -> None:
    data = tflite_path.read_bytes()
    with out_h.open("w", encoding="utf-8") as f:
        f.write("#pragma once\n#include <stdint.h>\n\n")
        f.write(f"// Auto-generated from: {tflite_path.name}\n")
        f.write(f"const unsigned char {var_name}[] = {{")
        for i, b in enumerate(data):
            if i % 12 == 0:
                f.write("\n  ")
            f.write(f"0x{b:02x}, ")
        f.write("\n};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")
    print(f"[OK] Wrote {out_h} ({len(data)} bytes)")

def main():
    ap = argparse.ArgumentParser(description="Convert ONNX -> (SavedModel) -> INT8 TFLite + Arduino model_data.h")
    ap.add_argument("--onnx", required=True, help="Path to ONNX file (e.g., best.onnx)")
    ap.add_argument("--out", default="esp32_export", help="Output folder")
    ap.add_argument("--rep", default="dataset/images/val", help="Representative images folder for INT8 calibration")
    ap.add_argument("--imgsz", type=int, default=96, help="Input image size (H=W=imgsz)")
    ap.add_argument("--channels", type=int, choices=[1,3], default=3, help="Input channels guess (3=RGB, 1=grayscale)")
    ap.add_argument("--rep_n", type=int, default=50, help="How many images to use for calibration")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    out_dir = Path(args.out).resolve()
    rep_dir = Path(args.rep).resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) ONNX -> onnx2tf output
    # NOTE: onnx2tf CLI should be available after pip install onnx2tf
    # It often creates tf_out/saved_model or direct tflite in output.
    tf_out = out_dir / "tf_out"
    tf_out.mkdir(parents=True, exist_ok=True)

    # Convert ONNX -> TF
    # Some models need options; we keep it minimal first.
    run(["onnx2tf", "-i", str(onnx_path), "-o", str(tf_out)])

    # If onnx2tf produced a .tflite directly, we can try to use it,
    # but for INT8 we prefer making our own via TF Lite converter.
    direct_tflite = find_first_tflite(tf_out)

    # 2) Build INT8 TFLite from SavedModel if present
    # onnx2tf typically creates: tf_out/saved_model
    saved_model_dir = tf_out / "saved_model"
    if not saved_model_dir.exists():
        # Some onnx2tf versions output to a different folder name.
        # Try to locate a SavedModel folder containing saved_model.pb
        candidates = [p.parent for p in tf_out.rglob("saved_model.pb")]
        if candidates:
            saved_model_dir = candidates[0]
        else:
            if direct_tflite:
                print("[WARN] SavedModel not found; only direct .tflite exists:", direct_tflite)
                print("       Will still generate Arduino header from the direct .tflite (NOT INT8-quantized).")
                tflite_path = out_dir / "model.tflite"
                tflite_path.write_bytes(direct_tflite.read_bytes())
                write_c_array(tflite_path, out_dir / "model_data.h")
                print("\nNext: Use Arduino TFLite Micro sketch to load model_data.h")
                return
            raise RuntimeError("SavedModel not found and no .tflite produced. onnx2tf output unexpected.")

    print("[OK] SavedModel:", saved_model_dir)

    # Import tensorflow here (after onnx2tf) to keep errors clearer
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset (INT8 calibration)
    rep_batch = load_rep_images(
        rep_dir=rep_dir,
        n=args.rep_n,
        input_hw=(args.imgsz, args.imgsz),
        channels=args.channels,
    )

    def rep_dataset_gen():
        for i in range(rep_batch.shape[0]):
            # TFLite converter expects list of input arrays
            yield [rep_batch[i:i+1]]

    converter.representative_dataset = rep_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print("[INFO] Converting to INT8 TFLite...")
    tflite_model = converter.convert()

    tflite_path = out_dir / "best_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"[OK] Wrote {tflite_path} ({tflite_path.stat().st_size} bytes)")

    # 3) Write Arduino header
    out_h = out_dir / "model_data.h"
    write_c_array(tflite_path, out_h)

    print("\nDONE.")
    print("Outputs:")
    print(" -", tflite_path)
    print(" -", out_h)
    print("\nIf conversion fails with 'unsupported op', that means YOLO ops are not MCU-friendly;")
    print("you can still submit Q2a as 'complete module design + deployment limitation analysis' and use Q5 (FOMO) for ESP32 detection.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", str(e))
        sys.exit(1)
