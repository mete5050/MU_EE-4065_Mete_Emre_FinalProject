import os, random, shutil
from pathlib import Path

# Ayarlar
RAW_DIR = Path("raw_images")
OUT_DIR = Path("dataset")
TRAIN_RATIO = 0.8
SEED = 42

# YOLO label: class x_center y_center width height (0-1)
# Tam kare bbox:
FULL_BOX = "0.5 0.5 1.0 1.0"

def class_from_filename(fname: str) -> int:
    # Örn: "7_012.jpg" -> 7
    first = fname.strip()[0]
    if first.isdigit():
        return int(first)
    raise ValueError(f"Dosya adı rakamla başlamıyor: {fname}")

def main():
    random.seed(SEED)
    assert RAW_DIR.exists(), f"{RAW_DIR} yok. Görselleri buraya koy."

    # Çıkış klasörleri
    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images += list(RAW_DIR.glob(ext))
    images = sorted(images)
    assert len(images) > 0, "raw_images içinde görüntü bulunamadı."

    random.shuffle(images)
    n_train = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    def write_split(img_list, split):
        for img_path in img_list:
            cls = class_from_filename(img_path.name)
            # görüntüyü kopyala
            dst_img = OUT_DIR / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # label yaz
            label_path = OUT_DIR / "labels" / split / (img_path.stem + ".txt")
            label_path.write_text(f"{cls} {FULL_BOX}\n", encoding="utf-8")

    write_split(train_imgs, "train")
    write_split(val_imgs, "val")

    # data.yaml
    yaml = """path: dataset
train: images/train
val: images/val

names:
  0: zero
  1: one
  2: two
  3: three
  4: four
  5: five
  6: six
  7: seven
  8: eight
  9: nine
"""
    (OUT_DIR / "data.yaml").write_text(yaml, encoding="utf-8")

    print("Bitti!")
    print(f"Toplam: {len(images)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")
    print("Şimdi eğitim için: yolo detect train model=yolo11n.pt data=dataset/data.yaml imgsz=96 epochs=100")

if __name__ == "__main__":
    main()
