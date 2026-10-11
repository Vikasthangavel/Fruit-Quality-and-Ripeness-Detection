import argparse
import json
import os
from pathlib import Path

# Reduce TensorFlow verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Fruit Quality from an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="fruit_quality_model.keras", help="Path to trained model")
    parser.add_argument("--classes", type=str, default="class_names.json", help="Path to class names JSON")
    parser.add_argument("--output", type=str, default=None, help="Optional output path to save annotated image")
    parser.add_argument("--show", action="store_true", help="Display prediction in OpenCV window")
    return parser.parse_args()


def load_model_and_classes(model_path, classes_path):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}. Please run train.py first.")
    if not Path(classes_path).is_file():
        raise FileNotFoundError(f"Classes file not found: {classes_path}. Please run train.py first.")

    model = tf.keras.models.load_model(model_path)
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


def predict_image(image_path, model, class_names):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image using OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image file: {image_path}")

    # Prepare input for model
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, IMAGE_SIZE)
    tensor = np.expand_dims(resized_img.astype(np.float32), axis=0)

    # Predict
    preds = model.predict(tensor, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    best_label = class_names[best_idx]
    best_conf = float(preds[best_idx])

    # Parse Fruit and Quality
    parts = best_label.rsplit("_", 1)
    fruit = parts[0].replace("_", " ").title()
    quality = parts[1].upper() if len(parts) == 2 else "UNKNOWN"

    return {
        "label": best_label,
        "fruit": fruit,
        "quality": quality,
        "confidence": best_conf,
        "all_predictions": {class_names[i]: float(preds[i]) for i in range(len(class_names))},
        "original_image": img,
    }


def draw_annotation(image, result):
    img = image.copy()
    h, w = img.shape[:2]

    fruit = result["fruit"]
    quality = result["quality"]
    conf = result["confidence"]

    # Color: Green for GOOD, Red for BAD
    if quality == "GOOD":
        color = (0, 200, 0)
        status_text = f"{fruit} - FRESH / GOOD"
    else:
        color = (0, 0, 230)
        status_text = f"{fruit} - ROTTEN / BAD"

    # Overlay banner at top
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Border frame
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), color, 3)

    # Text details
    cv2.putText(img, status_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
    cv2.putText(
        img,
        f"Confidence: {conf:.1%}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return img


def main():
    args = parse_args()
    model, class_names = load_model_and_classes(args.model, args.classes)
    result = predict_image(args.image, model, class_names)

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"File       : {args.image}")
    print(f"Fruit      : {result['fruit']}")
    print(f"Quality    : {result['quality']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print("\nAll Class Probabilities:")
    for cls, score in sorted(result["all_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cls:15s}: {score * 100:6.2f}%")
    print("=" * 50)

    annotated = draw_annotation(result["original_image"], result)

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"Annotated result saved to: {args.output}")

    if args.show:
        cv2.imshow("Fruit Quality Prediction", annotated)
        print("Press any key in the window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
