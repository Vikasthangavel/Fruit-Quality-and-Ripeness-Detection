# 🍎 Fruit Quality AI Detector

Deep Learning Computer Vision project for automated real-time fruit classification and freshness/quality assessment (Good vs. Bad / Fresh vs. Rotten) using **TensorFlow 2**, **MobileNetV2 Transfer Learning**, **OpenCV**, and **Streamlit**.

---

## 📁 Dataset Structure

The project automatically detects the local `dataset/` folder containing images categorized into 6 classes:
- `Apple_Good` (~1,134 images)
- `Apple_Bad` (~1,141 images)
- `Banana_Good` (~1,113 images)
- `Banana_Bad` (~1,087 images)
- `Lime_Good` (~1,094 images)
- `Lime_Bad` (~1,085 images)

Total: **~6,654 images**

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Train the MobileNetV2 classifier with data augmentation (horizontal flip, slight rotation, zoom):
```bash
python train.py
```
Optional training flags:
```bash
python train.py --epochs 15 --batch_size 32 --lr 0.001
```
During training:
- Best weights are automatically saved to `fruit_quality_model.keras`
- Class labels are exported to `class_names.json`
- Performance curves (Loss & Accuracy) are saved to `training_history.png`

---

### 3. Run Real-Time Webcam Detection
Detect fruit quality live through your webcam:
```bash
python webcam_detect.py
```
**Controls**:
- `Q`: Quit the live camera feed
- `S`: Save a timestamped snapshot screenshot
- Optional flags:
  ```bash
  python webcam_detect.py --camera 0 --threshold 0.65
  ```

---

### 4. Test Single Images
To evaluate any single photo or test image:
```bash
python predict.py --image dataset/Apple_Good/20190809_115439.jpg --show
```
To save the annotated result:
```bash
python predict.py --image dataset/Banana_Bad/20190809_121543.jpg --output result.jpg
```

---

### 5. Launch the Web Application (Streamlit GUI)
Run the interactive browser dashboard:
```bash
streamlit run app.py
```
Features in the Web App:
- 📁 **File Upload**: Drag and drop any fruit image to get instant analysis and class confidence charts.
- 🔍 **Sample Browser**: Pick random sample images directly from the dataset.
- 📸 **Webcam Snapshot**: Take a photo directly in the browser and get quality predictions.

---

## 🧠 Model Architecture

- **Backbone**: MobileNetV2 pretrained on ImageNet (lightweight and optimal for real-time edge/webcam inference)
- **Input Dimensions**: `(224, 224, 3)`
- **Data Augmentation**: RandomFlip, RandomRotation(0.08), RandomZoom(0.10)
- **Classification Head**: GlobalAveragePooling2D + Dropout(0.30) + Dense Softmax (6 classes)
- **Optimizer**: Adam (learning rate = 0.001) with `ReduceLROnPlateau` and `EarlyStopping`
