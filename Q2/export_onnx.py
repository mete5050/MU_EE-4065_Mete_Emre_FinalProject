from ultralytics import YOLO
m = YOLO("runs/detect/train/weights/best.pt")
m.export(format="onnx", imgsz=96)
