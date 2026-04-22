# ============================================================
# Emotion Detection with MobileNetV2 + TFLite Export
# Classes: neutral, happy, sad  (from AffectNet via kagglehub)
# ============================================================

# ============================================================
# 1. Imports
# ============================================================
import os
import json
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# 2. Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 3. Paths
# ============================================================
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))

import kagglehub
data_path = kagglehub.dataset_download('minhtmnguyntrn/affectnet-dataset')
data_root = os.path.join(data_path, "data") if os.path.isdir(os.path.join(data_path, "data")) else data_path

# Inspect top-level structure so we can confirm the folder layout
print("Top-level contents:", sorted(os.listdir(data_path)))
print("Using image root:", data_root)

# ============================================================
# 5. Configuration
# ============================================================
TARGET_SIZE      = 160
IMAGES_PER_CLASS = 3000

# Kaggle AffectNet export uses numeric folder ids.
# Standard AffectNet mapping:
# 0=neutral, 1=happy, 2=sad, 3=surprise, 4=fear, 5=disgust, 6=anger, 7=contempt
label_map = {
    '0': 'neutral',
    '1': 'happy',
    '2': 'sad',
}
SELECTED_CLASSES = set(label_map.keys())

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

BATCH_SIZE       = 32
HEAD_EPOCHS      = 12
FINE_TUNE_EPOCHS = 8
HEAD_LR          = 1e-3
FINE_TUNE_LR     = 5e-6   # reduced to avoid destabilising fine-tune restart

# MobileNetV2 width multiplier — keep small for ESP32 / TFLite
ALPHA        = 0.5
# Freeze all layers up to this index during fine-tuning (higher = fewer trainable layers)
FINE_TUNE_AT = 125        # freezing more layers stabilises fine-tuning

# ============================================================
# 6. Output directories
# ============================================================
output_path  = os.path.join(NOTEBOOK_DIR, "emotion_dataset_mobilenet")
MODEL_DIR    = os.path.join(NOTEBOOK_DIR, "generated_mobilenet")

for d in [output_path, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

HEAD_MODEL_PATH  = os.path.join(MODEL_DIR, "emotion_mobilenet_head.keras")
FINE_MODEL_PATH  = os.path.join(MODEL_DIR, "emotion_mobilenet_finetuned.keras")
TFLITE_PATH      = os.path.join(MODEL_DIR, "emotion_mobilenet.tflite")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
MISCLASSIFIED_DIR = os.path.join(MODEL_DIR, "inspect_misclassified")

print("Output path:", output_path)
print("Model dir:  ", MODEL_DIR)

for split_name in ["train", "val", "test"]:
    split_dir = os.path.join(output_path, split_name)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

if os.path.exists(MISCLASSIFIED_DIR):
    shutil.rmtree(MISCLASSIFIED_DIR)
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

# ============================================================
# 7. Find valid images per class
# ============================================================
valid_images_per_class = {}

for class_folder in sorted(os.listdir(data_root)):
    if class_folder not in SELECTED_CLASSES:
        print(f"Skipping class '{class_folder}'")
        continue

    class_path = os.path.join(data_root, class_folder)
    if not os.path.isdir(class_path):
        continue

    valid = []
    for img_file in os.listdir(class_path):
        img_full_path = os.path.join(class_path, img_file)
        try:
            with Image.open(img_full_path) as img:
                w, h = img.size
                if w >= TARGET_SIZE and h >= TARGET_SIZE:
                    valid.append(img_file)
        except Exception:
            pass

    valid_images_per_class[class_folder] = valid
    print(f"Class {class_folder} ({label_map[class_folder]}): {len(valid)} valid images")

# ============================================================
# 8. Sample images per class
# ============================================================
selected_per_class = {}

for class_folder, valid in valid_images_per_class.items():
    if len(valid) < IMAGES_PER_CLASS:
        print(f"Class {class_folder} ({label_map[class_folder]}): "
              f"only {len(valid)} available, taking all")
        selected_per_class[class_folder] = valid
    else:
        selected_per_class[class_folder] = random.sample(valid, IMAGES_PER_CLASS)
        print(f"Class {class_folder} ({label_map[class_folder]}): "
              f"sampled {IMAGES_PER_CLASS} images")

# ============================================================
# 9. Split into train / val / test
# ============================================================
splits_per_class = {}

for class_folder, selected in selected_per_class.items():
    random.shuffle(selected)
    n       = len(selected)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits_per_class[class_folder] = {
        'train': selected[:n_train],
        'val':   selected[n_train:n_train + n_val],
        'test':  selected[n_train + n_val:]
    }

    s = splits_per_class[class_folder]
    print(f"Class {label_map[class_folder]}: "
          f"train={len(s['train'])}  val={len(s['val'])}  test={len(s['test'])}")

# ============================================================
# 10. Resize and save dataset
# ============================================================
for class_folder, splits in splits_per_class.items():
    emotion_name = label_map[class_folder]
    class_path   = os.path.join(data_root, class_folder)

    for split_name, files in splits.items():
        out_dir = os.path.join(output_path, split_name, emotion_name)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        for img_file in files:
            img_full_path = os.path.join(class_path, img_file)
            out_img_path  = os.path.join(out_dir, img_file)

            try:
                with Image.open(img_full_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                    img.save(out_img_path, 'JPEG', quality=95)
                    saved += 1
            except Exception as e:
                print(f"  Error on {img_file}: {e}")

        print(f"  {split_name}/{emotion_name}: saved {saved} images")

# ============================================================
# 11. Final dataset summary
# ============================================================
print("\nFinal dataset summary:")
for split in ['train', 'val', 'test']:
    split_path  = os.path.join(output_path, split)
    split_total = 0
    print(f"\n{split}/")
    if not os.path.exists(split_path):
        continue
    for emotion in sorted(os.listdir(split_path)):
        emotion_path = os.path.join(split_path, emotion)
        if os.path.isdir(emotion_path):
            count = len(os.listdir(emotion_path))
            split_total += count
            print(f"  {emotion}: {count} images")
    print(f"  subtotal: {split_total}")

# ============================================================
# 12. Load datasets with Keras
# ============================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_path, "train"),
    image_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_path, "val"),
    image_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_path, "test"),
    image_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("\nClasses:", class_names)
print("Number of classes:", num_classes)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ============================================================
# 13. Compute class weights (handles class imbalance)
# ============================================================
y_train = []
for _, labels in train_ds.unbatch():
    y_train.append(int(labels.numpy()))

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=np.array(y_train)
)
class_weight_dict = dict(enumerate(class_weights_array))
print("\nClass weights:", class_weight_dict)

# ============================================================
# 14. Data augmentation — stronger for face/emotion robustness
# ============================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.04, 0.04),
    tf.keras.layers.RandomBrightness(0.1),   # lighting variation
    tf.keras.layers.RandomContrast(0.1),     # expression contrast
], name="augmentation")

# ============================================================
# 15. Build MobileNetV2 transfer-learning model
#     Head: two small dense layers (adds ~20K params — negligible for TFLite)
# ============================================================
def build_mobilenet_model(input_size, num_classes, alpha=0.5, dropout_rate=0.3):
    base_model = MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=alpha
    )
    base_model.trainable = False

    inputs = Input(shape=(input_size, input_size, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)             # scale pixels to [-1, 1]
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model, base_model


model, base_model = build_mobilenet_model(
    input_size=TARGET_SIZE,
    num_classes=num_classes,
    alpha=ALPHA
)

# Label smoothing reduces overconfidence on noisy emotion labels
model.compile(
    optimizer=Adam(learning_rate=HEAD_LR),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 16. Phase 1 — Train classification head (base frozen)
# ============================================================
head_callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
    ModelCheckpoint(filepath=HEAD_MODEL_PATH, monitor='val_loss', save_best_only=True),
]

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=head_callbacks,
    class_weight=class_weight_dict,
)

# ============================================================
# 17. Phase 2 — Fine-tune upper layers of MobileNetV2
#     Only unfreeze layers above FINE_TUNE_AT to keep training stable.
#     Use a much lower LR to avoid destroying pretrained weights.
# ============================================================
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

fine_callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6),
    ModelCheckpoint(filepath=FINE_MODEL_PATH, monitor='val_loss', save_best_only=True),
]

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=fine_callbacks,
    class_weight=class_weight_dict,
)

# ============================================================
# 18. Plot training history
# ============================================================
def plot_history(history, title_prefix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     label='train acc')
    ax1.plot(history.history['val_accuracy'], label='val acc')
    ax1.set_title(f'{title_prefix} Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'],     label='train loss')
    ax2.plot(history.history['val_loss'], label='val loss')
    ax2.set_title(f'{title_prefix} Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

plot_history(history_head, "Head Training")
plot_history(history_fine, "Fine-Tuning")

# ============================================================
# 19. Load best fine-tuned model and evaluate
# ============================================================
best_model = tf.keras.models.load_model(FINE_MODEL_PATH)

test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"\nTest loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# ============================================================
# 20. Classification report + confusion matrix
#     Save misclassified samples for manual inspection.
# ============================================================
MAX_SAVED_MISCLASSIFIED_PER_PAIR = 40
y_true, y_pred = [], []
saved_misclassified_counts = {}

for batch_idx, (images, labels) in enumerate(test_ds):
    preds = best_model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(pred_labels)

    for img_idx, (image, true_idx, pred_idx, pred_probs) in enumerate(
        zip(images, labels.numpy(), pred_labels, preds)
    ):
        if true_idx == pred_idx:
            continue

        true_name = class_names[int(true_idx)]
        pred_name = class_names[int(pred_idx)]
        pair_name = f"{true_name}__pred_{pred_name}"
        pair_dir = os.path.join(MISCLASSIFIED_DIR, pair_name)
        os.makedirs(pair_dir, exist_ok=True)

        saved_count = saved_misclassified_counts.get(pair_name, 0)
        if saved_count >= MAX_SAVED_MISCLASSIFIED_PER_PAIR:
            continue

        confidence = float(pred_probs[pred_idx])
        image_uint8 = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8).numpy()
        out_name = (
            f"batch{batch_idx:03d}_img{img_idx:02d}"
            f"_true-{true_name}_pred-{pred_name}"
            f"_conf-{confidence:.2f}.jpg"
        )
        Image.fromarray(image_uint8).save(os.path.join(pair_dir, out_name), "JPEG", quality=95)
        saved_misclassified_counts[pair_name] = saved_count + 1

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45)
ax.set_yticklabels(class_names)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.show()

print("\nSaved misclassified samples:")
for pair_name in sorted(saved_misclassified_counts):
    print(f"  {pair_name}: {saved_misclassified_counts[pair_name]}")
print(f"Inspect folder: {MISCLASSIFIED_DIR}")
