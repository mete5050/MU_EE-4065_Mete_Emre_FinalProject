from ultralytics import YOLO

# Modeli yükle (ilk seferde indirir)
model = YOLO("yolo11n.pt")

# Eğitim
results = model.train(
    data="dataset/data.yaml",
    imgsz=96,
    epochs=100,
    batch=16,
    device="cpu",   # GPU varsa sonra "0" yaparız
)

print("Training done!")
print(results)
