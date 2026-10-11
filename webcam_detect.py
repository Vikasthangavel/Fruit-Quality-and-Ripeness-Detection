import argparse
import datetime
import json
import os
from pathlib import Path
import time

# Reduce TensorFlow verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Fruit Quality Detection using Webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold (default: 0.60)")
    parser.add_argument("--model", type=str, default="fruit_quality_model.keras", help="Path to trained model")
    parser.add_argument("--classes", type=str, default="class_names.json", help="Path to class names JSON")
    return parser.parse_args()


def load_model_and_classes(model_path, classes_path):
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Trained model '{model_path}' not found! Please run 'python train.py' first to train the model."
        )
    if not Path(classes_path).is_file():
        raise FileNotFoundError(
            f"Class mapping '{classes_path}' not found! Please run 'python train.py' first."
        )

    print(f"Loading model: {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} classes: {', '.join(class_names)}")
    return model, class_names


def predict_quality(model, class_names, crop):
    # Convert OpenCV BGR to RGB
    rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, IMAGE_SIZE)
    tensor = np.expand_dims(resized_image.astype(np.float32), axis=0)

    predictions = model.predict(tensor, verbose=0)[0]
    index = int(np.argmax(predictions))

    return class_names[index], float(predictions[index])


def main():
    args = parse_args()
    model, class_names = load_model_and_classes(args.model, args.classes)

    print(f"\nAttempting to connect to camera {args.camera}...")
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        # Fallback to standard backend
        camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open webcam (index {args.camera}). "
            "Please check camera permissions, plug in your webcam, or try '--camera 1'."
        )

    print("=" * 60)
    print("FRUIT QUALITY REAL-TIME DETECTION ACTIVE")
    print("Controls:")
    print("  [Q] - Quit application")
    print("  [S] - Save snapshot screenshot")
    print("=" * 60)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Warning: Unable to read webcam frame.")
                break

            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(current_time - prev_time, 1e-5))
            prev_time = current_time

            height, width = frame.shape[:2]

            # Central ROI box for fruit placement
            box_size = int(min(height, width) * 0.70)
            x1 = max(0, (width - box_size) // 2)
            y1 = max(0, (height - box_size) // 2)
            x2 = min(width, x1 + box_size)
            y2 = min(height, y1 + box_size)

            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                label, confidence = predict_quality(model, class_names, crop)

                # Decode the class label (e.g., Apple_Good -> Apple, GOOD)
                parts = label.rsplit("_", 1)
                fruit = parts[0].replace("_", " ").title()
                quality = parts[1].upper() if len(parts) == 2 else "UNKNOWN"

                if confidence >= args.threshold:
                    status_text = f"{fruit}: {quality}"
                    if quality == "GOOD":
                        color = (0, 220, 0)      # Bright Green
                    else:
                        color = (0, 40, 230)     # Bright Red
                else:
                    status_text = "Aligning Fruit (Uncertain)"
                    color = (0, 165, 255)         # Orange
            else:
                label, confidence = "Unknown", 0.0
                status_text = "No region"
                color = (200, 200, 200)

            # Draw semi-transparent header HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 90), (20, 20, 20), -1)
            cv2.rectangle(overlay, (0, height - 40), (width, height), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Draw central target bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Corner accents on bounding box
            corner_len = int(box_size * 0.12)
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 6)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 6)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 6)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 6)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 6)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 6)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 6)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 6)

            # Header text
            cv2.putText(
                frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, f"Confidence: {confidence:.1%}  |  FPS: {fps:.1f}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA
            )

            # Footer guide instructions
            cv2.putText(
                frame, "Place single fruit inside square | [Q] Quit | [S] Screenshot",
                (20, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
            )

            cv2.imshow("Fruit Quality Real-Time Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"Snapshot saved to: {snap_path}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")


if __name__ == "__main__":
    main()
