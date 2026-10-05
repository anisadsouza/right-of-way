from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 emergency vehicle classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/images"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--alpha", type=float, default=0.35, help="MobileNet width; 0.35 is suitable for Pi 4 (2 GB).")
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet", help="Use ImageNet transfer-learning weights or train offline from scratch.")
    parser.add_argument("--report", type=Path, default=Path("reports/vision_training_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    class_dirs = [path for path in args.data.iterdir() if path.is_dir()]
    if len(class_dirs) < 2:
        raise SystemExit("Put images in at least two labelled folders under data/images.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    labels = train_ds.class_names
    (args.out / "vision_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
        ]
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=(args.img_size, args.img_size, 3),
        include_top=False,
        weights=None if args.weights == "none" else "imagenet",
        alpha=args.alpha,
    )
    base.trainable = args.weights == "none"

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    loss, accuracy = model.evaluate(val_ds, verbose=0)

    keras_path = args.out / "vision_ev_mobilenet.keras"
    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    (args.out / "vision_ev_mobilenet.tflite").write_bytes(tflite)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "labels": labels,
        "image_size": args.img_size,
        "mobilenet_alpha": args.alpha,
        "initial_weights": args.weights,
        "epochs_completed": len(history.history["loss"]),
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    args.report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {keras_path} and TFLite model in {args.out}")


if __name__ == "__main__":
    main()
