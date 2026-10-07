"""
eval_vision.py

Loads the trained vision model and evaluates it per-class on the exact
same validation split train_vision.py used (same seed, same ratio),
so this is a true measurement, not a re-guess at the split.

Run from inside ev_recognition/, with your venv active:
    python eval_vision.py
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 192          # must match the --img-size used in your last training run
DATA_DIR = "data/images"
MODEL_PATH = "models/vision_ev_mobilenet.keras"

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    shuffle=False,
)

labels = val_ds.class_names
print("Classes:", labels)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []

for images, targets in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(targets.numpy())

print("\n--- Per-class results on the actual held-out validation set ---\n")
print(classification_report(y_true, y_pred, target_names=labels, digits=4))

print("Confusion matrix (rows = actual, columns = predicted):")
print(labels)
print(confusion_matrix(y_true, y_pred))