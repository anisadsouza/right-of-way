from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf


CLASS_NAMES = ["ambulance", "fire_truck", "normal", "police"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/scenes"))
    parser.add_argument("--out", type=Path, default=Path("models/scene_model"))
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--finetune-layers", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.35)
    return parser.parse_args()


def load_split(data_dir, split, img_size, batch_size, shuffle):
    folder = data_dir / split

    for name in CLASS_NAMES:
        if not (folder / name).is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder / name}")

    return tf.keras.utils.image_dataset_from_directory(
        folder,
        class_names=CLASS_NAMES,
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )


def calculate_class_weights(train_dir):
    counts = {
        i: len(list((train_dir / name).glob("*")))
        for i, name in enumerate(CLASS_NAMES)
    }

    if any(count == 0 for count in counts.values()):
        raise ValueError(f"A training class has no images: {counts}")

    total = sum(counts.values())
    weights = {
        i: math.sqrt(total / (len(CLASS_NAMES) * count))
        for i, count in counts.items()
    }

    print("\nTraining image counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts[i]} images, weight={weights[i]:.3f}")

    return weights


def evaluate_per_class(model, test_ds):
    probabilities = model.predict(test_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    actual = np.concatenate([labels.numpy() for _, labels in test_ds])
    confusion = tf.math.confusion_matrix(
        actual, predictions, num_classes=len(CLASS_NAMES)
    ).numpy()

    results = {}
    print("\n=== TEST SET RESULTS ===")
    print("Rows = actual class; columns = predicted class")
    print(confusion)

    for i, name in enumerate(CLASS_NAMES):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[i, :].sum()),
        }

        print(
            f"{name:12s} "
            f"precision={precision:.3f} "
            f"recall={recall:.3f} "
            f"F1={f1:.3f} "
            f"support={int(confusion[i, :].sum())}"
        )

    accuracy = float(np.mean(actual == predictions))
    print(f"\nOverall test accuracy: {accuracy:.4f}")

    return confusion.tolist(), results, accuracy


def main():
    args = parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(
            f"Output folder is not empty: {args.out}\n"
            "Choose a new output folder rather than overwriting a model."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(
        args.data, "train", args.img_size, args.batch_size, True
    )
    valid_ds = load_split(
        args.data, "valid", args.img_size, args.batch_size, False
    )
    test_ds = load_split(
        args.data, "test", args.img_size, args.batch_size, False
    )

    class_weights = calculate_class_weights(args.data / "train")

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ])

    base = tf.keras.applications.MobileNetV2(
        input_shape=(args.img_size, args.img_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=args.alpha,
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]

    print("\n=== PHASE 1: Train classifier head ===")
    history1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    history2 = None

    if args.finetune_epochs > 0:
        print("\n=== PHASE 2: Fine-tune MobileNetV2 ===")
        base.trainable = True

        for layer in base.layers[:-args.finetune_layers]:
            layer.trainable = False

        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history2 = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=args.finetune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )

    confusion, per_class, accuracy = evaluate_per_class(model, test_ds)

    keras_path = args.out / "scene_vision_mobilenet.keras"
    tflite_path = args.out / "scene_vision_mobilenet.tflite"

    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    tflite_path.write_bytes(tflite)

    (args.out / "vision_labels.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
    )

    report = {
        "classes": CLASS_NAMES,
        "image_size": args.img_size,
        "alpha": args.alpha,
        "test_accuracy": accuracy,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "class_weights": class_weights,
        "phase1_epochs": len(history1.history["loss"]),
        "finetune_epochs": (
            len(history2.history["loss"]) if history2 else 0
        ),
    }

    (args.out / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nTraining and testing complete.")
    print("TFLite model:", tflite_path)
    print("Labels:", args.out / "vision_labels.txt")
    print("Metrics:", args.out / "test_metrics.json")


if __name__ == "__main__":
    main()
