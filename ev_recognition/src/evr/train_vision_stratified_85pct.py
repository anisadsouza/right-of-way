from __future__ import annotations

import argparse
import json
import random
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


def build_datasets(data_dir: Path, img_size: int, batch_size: int, seed: int = 42, val_split: float = 0.2):
    """
    Builds train/validation datasets with a PER-CLASS stratified split, so every
    class is guaranteed to be represented proportionally in both sets, regardless
    of folder read order or class size. This replaces image_dataset_from_directory's
    built-in split, which was observed to take a contiguous slice of the
    class-ordered file list rather than a true shuffled split across all classes.
    """
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    labels = [p.name for p in class_dirs]
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    rng = random.Random(seed)

    for class_idx, class_dir in enumerate(class_dirs):
        files = sorted(str(p) for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_split))
        val_files = files[:n_val]
        train_files = files[n_val:]
        val_paths.extend(val_files)
        val_labels.extend([class_idx] * len(val_files))
        train_paths.extend(train_files)
        train_labels.extend([class_idx] * len(train_files))

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (img_size, img_size))
        return img, label

    def make_ds(paths, labs, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((paths, labs))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(train_paths, train_labels, shuffle=True)
    val_ds = make_ds(val_paths, val_labels, shuffle=False)

    print("Stratified split check (should be nonzero for every class on both sides):")
    for idx, name in enumerate(labels):
        print(f"  {name}: {train_labels.count(idx)} train, {val_labels.count(idx)} val")

    return train_ds, val_ds, labels


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    class_dirs = [path for path in args.data.iterdir() if path.is_dir()]
    if len(class_dirs) < 2:
        raise SystemExit("Put images in at least two labelled folders under data/images.")

    train_ds, val_ds, labels = build_datasets(args.data, args.img_size, args.batch_size)
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
    