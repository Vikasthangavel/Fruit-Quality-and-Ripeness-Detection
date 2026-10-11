import json
import os
from pathlib import Path
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Fruit Quality AI Detector",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "fruit_quality_model.keras"
CLASSES_PATH = "class_names.json"
IMAGE_SIZE = (224, 224)


@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).is_file():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    if not Path(CLASSES_PATH).is_file():
        return None
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict(image: Image.Image, model, class_names):
    # Ensure RGB
    rgb = image.convert("RGB")
    resized = rgb.resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)

    preds = model.predict(arr, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    best_label = class_names[best_idx]
    best_conf = float(preds[best_idx])

    parts = best_label.rsplit("_", 1)
    fruit = parts[0].replace("_", " ").title()
    quality = parts[1].upper() if len(parts) == 2 else "UNKNOWN"

    all_scores = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return fruit, quality, best_conf, all_scores


# UI Header
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #10B981, #059669); padding: 24px; border-radius: 12px; color: white; margin-bottom: 24px;">
        <h1 style="margin: 0; font-size: 2.2rem;">🍎 Fruit Quality AI Detector</h1>
        <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
            Real-Time Deep Learning Computer Vision for Fresh vs. Rotten Quality Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()
class_names = load_classes()

if model is None or class_names is None:
    st.warning("⚠️ **Trained model not found!**")
    st.info(
        """
        Please train the model first by running in your terminal:
        ```bash
        python train.py
        ```
        This will train the MobileNetV2 network on your `dataset/` images and generate `fruit_quality_model.keras`.
        """
    )
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings & Info")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.60, 0.05)
    st.markdown("---")
    st.subheader("Detected Fruit Classes")
    for cls in class_names:
        icon = "🍏" if "Good" in cls else "🍂"
        st.write(f"{icon} `{cls}`")
    st.markdown("---")
    st.caption("Powered by MobileNetV2 Transfer Learning & TensorFlow")

tab_upload, tab_samples, tab_cam = st.tabs(["📁 Upload Image", "🔍 Test Dataset Samples", "📸 Webcam Snapshot"])


def display_results(img, fruit, quality, conf, all_scores):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Analyzed Fruit Image", use_container_width=True)

    with col2:
        is_good = quality == "GOOD"
        is_confident = conf >= confidence_threshold

        if not is_confident:
            badge_color = "#F59E0B"
            badge_text = "UNCERTAIN / LOW CONFIDENCE"
        elif is_good:
            badge_color = "#10B981"
            badge_text = f"{fruit.upper()} — FRESH / GOOD QUALITY"
        else:
            badge_color = "#EF4444"
            badge_text = f"{fruit.upper()} — ROTTEN / BAD QUALITY"

        st.markdown(
            f"""
            <div style="background-color: {badge_color}; color: white; padding: 16px; border-radius: 8px; font-weight: bold; font-size: 1.25rem; text-align: center; margin-bottom: 16px;">
                {badge_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(label="Predicted Fruit", value=fruit)
        st.metric(label="Quality Verdict", value=quality, delta="Healthy" if is_good else "-Defective")
        st.metric(label="Prediction Confidence", value=f"{conf:.1%}")

        st.subheader("Class Probability Distribution")
        for cls, prob in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{cls}**: `{prob:.1%}`")
            st.progress(float(prob))


with tab_upload:
    uploaded = st.file_uploader("Upload a fruit photo (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded)
        fruit, quality, conf, scores = predict(image, model, class_names)
        display_results(image, fruit, quality, conf, scores)

with tab_samples:
    st.write("Pick a sample image from the local `dataset` directory to test instantly:")
    dataset_path = Path("dataset")
    if dataset_path.is_dir():
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        selected_class = st.selectbox("Select Class Folder", [d.name for d in class_dirs])
        if selected_class:
            images = list((dataset_path / selected_class).glob("*.*"))
            if images:
                col_btn, _ = st.columns([1, 3])
                with col_btn:
                    if st.button("🎲 Pick Random Sample"):
                        st.session_state["sample_img"] = str(random.choice(images))
                if "sample_img" not in st.session_state or not Path(st.session_state["sample_img"]).exists():
                    st.session_state["sample_img"] = str(images[0])

                sample_path = st.session_state["sample_img"]
                st.caption(f"Testing image: `{sample_path}`")
                image = Image.open(sample_path)
                fruit, quality, conf, scores = predict(image, model, class_names)
                display_results(image, fruit, quality, conf, scores)
    else:
        st.info("Dataset directory not found locally.")

with tab_cam:
    st.write("Capture a frame from your webcam:")
    camera_file = st.camera_input("Take a photo of a fruit")
    if camera_file:
        image = Image.open(camera_file)
        fruit, quality, conf, scores = predict(image, model, class_names)
        display_results(image, fruit, quality, conf, scores)
