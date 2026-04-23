#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil

# Force CPU before TF initializes
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.keras.backend.clear_session()
tf.config.set_visible_devices([], 'GPU')

# Delete any saved checkpoints from previous runs
for f in ['best_model.keras', 'best_model_v2.keras']:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted {f}")

# Clear TF's internal caches
tf.compat.v1.reset_default_graph()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("TF version:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
print("Ready — clean slate confirmed")


# In[2]:


# Cell 2
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 150
NUM_CLASSES = 3
CLASSES = ['surprise', 'happy', 'sad']
DATA_DIR = 'emotion_dataset/'


# In[3]:


# Cell 3
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


# In[4]:


# Cell 4 — verify
# for images, labels in train_ds.take(1):
#     print("dtype:", images.dtype)
#     print("min:", images.numpy().min(), "max:", images.numpy().max())
#     print("labels:", labels.numpy()[:8])
#     fig, axes = plt.subplots(1, 4, figsize=(12, 3))
#     for i in range(4):
#         axes[i].imshow(images[i].numpy().astype('uint8'))
#         axes[i].set_title(CLASSES[labels[i]])
#         axes[i].axis('off')
#     plt.show()


# In[5]:


# Cell 5
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)


# In[6]:


tf.keras.backend.clear_session()

reg = tf.keras.regularizers.l2(1e-4)

def build_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding='same', kernel_regularizer=reg)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

model = build_cnn((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
#model.summary()


# In[7]:


# Cell 7
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint('best_model_v2.keras', save_best_only=True)
]
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'], label='train')
axes[0].plot(history.history['val_accuracy'], label='val')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[1].plot(history.history['loss'], label='train')
axes[1].plot(history.history['val_loss'], label='val')
axes[1].set_title('Loss')
axes[1].legend()
plt.tight_layout()
plt.savefig('training_curves.png')  # save instead of show
plt.close()

# Evaluation
y_pred = np.argmax(model.predict(test_ds), axis=1)
y_true = np.concatenate([y for _, y in test_ds])

print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_ylabel('True')
ax.set_xlabel('Predicted')
ax.set_title('Confusion matrix — test set')
plt.tight_layout()
plt.savefig('confusion_matrix.png')  # save instead of show
plt.close()

print("Done! Results saved to training_curves.png and confusion_matrix.png")




