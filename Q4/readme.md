# EE4065 – Final Project  
## Question 4: Multi-Model Handwritten Digit Recognition on ESP32-CAM

This project implements **handwritten digit recognition** on an **ESP32-CAM** module using **multiple lightweight CNN architectures** and a **model fusion (ensemble) strategy**, as required in **Question 4** of the EE4065 Final Project.

The system captures real-time images using the ESP32-CAM camera, preprocesses them on-device, runs inference using different CNN models, and merges their outputs to obtain a final prediction. A web-based dashboard is provided for live visualization and interaction.

---

## 📌 Project Objectives

- Implement **multiple CNN models** for handwritten digit recognition
- Deploy all models on **ESP32-CAM** using **TensorFlow Lite Micro**
- Compare model performance in terms of:
  - Prediction accuracy
  - Confidence
  - Inference time
- Apply **model fusion (ensemble learning)** to improve robustness
- Provide a **real-time web interface** for demonstration

---

## 🧠 Implemented CNN Architectures

The following models are used in this project:

| Model | Purpose |
|------|--------|
| **SqueezeNet (Mini)** | Lightweight baseline model |
| **MobileNetV2 (Mini)** | Accuracy–efficiency tradeoff |
| **ResNet-8** | Deeper architecture with residual connections |
| **EfficientNet (Mini)** | Efficient scaling-based CNN |
| **Ensemble Model** | Fusion of all models above |

Each model is trained separately and converted to **quantized `.tflite` format** for embedded deployment.

---

## 🧩 System Architecture

1. **Image Capture**
   - ESP32-CAM captures RGB frames (QVGA, RGB565)

2. **Preprocessing (On-device)**
   - Center crop
   - Grayscale conversion
   - Contrast stretching
   - Adaptive thresholding (FOMO-style)
   - Morphological dilation
   - Bounding box extraction
   - Resize to **32×32**
   - Quantization for TFLite input

3. **Inference**
   - TensorFlow Lite Micro interpreter
   - One model at a time (memory-efficient)
   - Inference time measured per model

4. **Fusion (Ensemble)**
   - Softmax probability averaging
   - Final decision from averaged probabilities

5. **Visualization**
   - Live camera feed
   - Preprocessed input image
   - Prediction results
   - Confidence and inference time
   - Model comparison table

---

## ⚙️ Hardware Requirements

- ESP32-CAM (AI-Thinker module)
- USB-to-TTL programmer
- Stable 5V power supply
- Wi-Fi connection (or AP mode)

---

## 💻 Software Requirements

- Arduino IDE
- ESP32 board support package
- TensorFlow Lite Micro (ESP32-compatible)
- Python 3.8+ (for training and conversion)
- TensorFlow / Keras
- NumPy, OpenCV

---



## 🧪 Model Training (Python)

Each model is trained using grayscale handwritten digit images (MNIST-style or custom dataset).

General training pipeline:
- Input size: **32×32×3**
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Quantization-aware training or post-training quantization
- Export to `.tflite`

All trained models are merged into a single `model_data.h` file for embedded deployment.

---

## 🔀 Model Fusion Strategy (Ensemble)

The ensemble model works as follows:

1. Run inference using **all four CNN models**
2. Extract softmax probabilities for each digit (0–9)
3. Compute the **average probability** for each class
4. Select the digit with the highest average probability

This improves robustness against:
- Lighting changes
- Writing variations
- Individual model errors

---

## 🌐 Web Interface Features

- Live camera feed
- Processed neural network input preview
- Model selection buttons
- Ensemble inference button
- Prediction confidence visualization
- Inference time benchmarking
- Probability bar graphs

The interface is served directly from the ESP32-CAM using an HTTP server.

---

## 🚀 How to Run

1. Open `ESP32_YOLO_Web.ino` in Arduino IDE
2. Select board: **AI Thinker ESP32-CAM**
3. Configure Wi-Fi credentials in the code
4. Upload the sketch
5. Open Serial Monitor to get the IP address
6. Access the web interface via browser

---

## 📊 Performance Notes

- All models are optimized to fit within ESP32-CAM memory limits
- PSRAM is used for tensor arena and preprocessing buffers
- Models are loaded dynamically to reduce memory usage
- Typical inference time: **tens to hundreds of milliseconds**

---

## ✅ Conclusion

This project demonstrates that **multi-model CNN inference and ensemble learning** can be successfully implemented on a **resource-constrained embedded platform** such as ESP32-CAM. By combining efficient architectures and intelligent preprocessing, reliable handwritten digit recognition is achieved in real time.



