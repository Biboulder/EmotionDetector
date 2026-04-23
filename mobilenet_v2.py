import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.clear_session()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Config ──────────────────────────────────────────────────
IMG_SIZE         = 96
BATCH_SIZE       = 32
CLASSES          = ['surprise', 'happy', 'sad']
NUM_CLASSES      = len(CLASSES)
DATA_DIR         = 'emotion_dataset/'
MODEL_DIR        = 'generated_mobilenet/'
ALPHA            = 0.5   # MobileNetV2 width multiplier — smaller = faster on ESP32
HEAD_EPOCHS      = 30
FINETUNE_EPOCHS  = 50
HEAD_LR          = 1e-3
FINETUNE_LR      = 1e-5 
FINE_TUNE_AT     = 60   # freeze layers below this index during fine-tuning

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data loading ─────────────────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_names=CLASSES,
    shuffle=True,
    seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'val'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_names=CLASSES,
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_names=CLASSES,
    shuffle=False
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).map(preprocess).prefetch(AUTOTUNE)
val_ds   = val_ds.map(preprocess).prefetch(AUTOTUNE)
test_ds  = test_ds.map(preprocess).prefetch(AUTOTUNE)

# ── Model ────────────────────────────────────────────────────
def build_mobilenet(input_size, num_classes, alpha):
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)  # correct preprocessing
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs), base_model

model, base_model = build_mobilenet(IMG_SIZE, NUM_CLASSES, ALPHA)
model.summary()

# ── Phase 1: Train head only ──────────────────────────────────
print("\n=== Phase 1: Training head ===")
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=HEAD_LR, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
head_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'head.keras'), save_best_only=True)
]

history_head = model.fit(
    train_ds, validation_data=val_ds,
    epochs=HEAD_EPOCHS, callbacks=head_callbacks
)

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'head.keras'))

# ── Phase 2: Fine-tune upper layers ──────────────────────────
print("\n=== Phase 2: Fine-tuning ===")
base_model = model.layers[1]
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False
# Keep BatchNorm frozen during fine-tuning
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=FINETUNE_LR, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.6, patience=6, min_lr=1e-7),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'finetuned.keras'), save_best_only=True)
]

history_fine = model.fit(
    train_ds, validation_data=val_ds,
    epochs=FINETUNE_EPOCHS, callbacks=fine_callbacks
)

# ── Evaluation ───────────────────────────────────────────────
print("\n=== Evaluation ===")
y_pred = np.argmax(model.predict(test_ds), axis=1)
y_true = np.concatenate([y for _, y in test_ds])

print(classification_report(y_true, y_pred, target_names=CLASSES))

# Save curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
all_acc = history_head.history['accuracy'] + history_fine.history['accuracy']
all_val = history_head.history['val_accuracy'] + history_fine.history['val_accuracy']
all_loss = history_head.history['loss'] + history_fine.history['loss']
all_val_loss = history_head.history['val_loss'] + history_fine.history['val_loss']

axes[0].plot(all_acc, label='train')
axes[0].plot(all_val, label='val')
axes[0].axvline(x=len(history_head.history['accuracy']), 
                color='gray', linestyle='--', label='fine-tune start')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[1].plot(all_loss, label='train')
axes[1].plot(all_val_loss, label='val')
axes[1].axvline(x=len(history_head.history['loss']), 
                color='gray', linestyle='--', label='fine-tune start')
axes[1].set_title('Loss')
axes[1].legend()
plt.tight_layout()
plt.savefig('mobilenet_curves.png')
plt.close()

# Save confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('MobileNetV2 — Confusion matrix')
plt.tight_layout()
plt.savefig('mobilenet_confusion.png')
plt.close()

# Save model
model.save(os.path.join(MODEL_DIR, 'final_model.keras'))
print("Done! Saved to", MODEL_DIR)