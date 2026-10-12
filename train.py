import argparse
import json
import os
from pathlib import Path

# Reduce TensorFlow verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ---------------- CONFIGURATION ----------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Fruit Quality Detection Model with MobileNetV2")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to dataset directory containing class subfolders (e.g., 'dataset' or 'fruit_dataset')",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--model_path", type=str, default="fruit_quality_model.keras", help="Path to save model")
    parser.add_argument("--classes_path", type=str, default="class_names.json", help="Path to save class mapping")
    return parser.parse_args()


def find_dataset_dir(user_specified_dir):
    if user_specified_dir:
        p = Path(user_specified_dir)
        if p.is_dir():
            return str(p)
        raise FileNotFoundError(f"Specified dataset directory not found: {user_specified_dir}")

    # Auto-detection
    candidates = ["dataset", "fruit_dataset", "../dataset"]
    for c in candidates:
        p = Path(c)
        if p.is_dir() and any(p.iterdir()):
            return str(p)

    raise FileNotFoundError(
        "Could not find dataset folder. Please specify using --dataset_dir <path>."
    )


def plot_history(history, output_path="training_history.png"):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2563EB", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#16A34A", linewidth=2)
    plt.title("Model Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#DC2626", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#F59E0B", linewidth=2)
    plt.title("Model Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Training history plot saved to: {output_path}")


def main():
    args = parse_args()
    dataset_dir = find_dataset_dir(args.dataset_dir)
    image_size = (224, 224)
    batch_size = args.batch_size
    epochs = args.epochs
    seed = 42

    print("=" * 60)
    print("FRUIT QUALITY CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset path: {dataset_dir}")
    print(f"Batch size  : {batch_size}")
    print(f"Image size  : {image_size}")
    print(f"Epochs      : {epochs}")
    print(f"Model target: {args.model_path}")
    print(f"Classes file: {args.classes_path}")
    print("=" * 60)

    # ---------------- LOAD DATA ---------------------
    print("\n[1/5] Loading and splitting dataset (80% Train, 20% Val)...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.20,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.20,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"\nDetected {num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}")

    if num_classes < 2:
        raise ValueError("At least two classes are required to train a classification model.")

    # Save class mapping for webcam and inference predictions
    with open(args.classes_path, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=4)
    print(f"\nSaved class mapping to: {args.classes_path}")

    # Optimize data pipeline
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    # ---------------- BUILD MODEL ------------------
    print("\n[2/5] Building MobileNetV2 Transfer Learning Architecture...")

    augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="augmentation",
    )

    # MobileNetV2 expects normalized inputs [-1, 1]
    # Rescaling handles: (x / 127.5) - 1.0 seamlessly inside the computation graph
    rescaling = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    x = augmentation(inputs)
    x = rescaling(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.30, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="FruitQualityMobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ---------------- TRAIN MODEL -------------------
    print("\n[3/5] Starting Model Training...")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ---------------- EVALUATE MODEL ---------------
    print("\n[4/5] Evaluating Best Model...")
    best_model = tf.keras.models.load_model(args.model_path)
    val_loss, val_acc = best_model.evaluate(val_ds, verbose=1)

    print("\n" + "=" * 60)
    print("TRAINING & VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Validation Loss    : {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Trained Model saved: {args.model_path}")
    print(f"Class names saved  : {args.classes_path}")
    print("=" * 60)

    print("\n[5/5] Generating training curves plot...")
    plot_history(history, output_path="training_history.png")

    print("\nAll steps completed successfully!")


if __name__ == "__main__":
    main()
