"""
Step 9-10: TFLite conversion, INT8 quantization, and C header export.

Outputs:
  generated_mobilenet/emotion_mobilenet_f32.tflite   - Float32 TFLite model
  generated_mobilenet/emotion_mobilenet_int8.tflite  - INT8 quantized TFLite model
  esp32/main/model.h                                 - C header with defines
  esp32/main/model.c                                 - C source with model binary

Usage:
  python convert_and_export.py <model.keras>
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    print("Usage: python convert_and_export.py <model.keras>")
    sys.exit(1)

model_arg = sys.argv[1]
MODEL_PATH = model_arg if os.path.isabs(model_arg) else os.path.join(SCRIPT_DIR, model_arg)
DATASET_DIR      = os.path.join(SCRIPT_DIR, "emotion_dataset")
GEN_DIR          = os.path.join(SCRIPT_DIR, "generated_mobilenet")
CLASS_NAMES_PATH = os.path.join(GEN_DIR, "class_names.json")
MODEL_H_PATH     = os.path.join(SCRIPT_DIR, "esp32", "main", "model.h")
MODEL_C_PATH     = os.path.join(SCRIPT_DIR, "esp32", "main", "model.c")

# Real-world ESP camera images used for INT8 calibration (representative dataset).
# These folders contain images captured directly from the XIAO ESP32-S3 camera.
REALWORLD_DIRS   = [
    os.path.join(SCRIPT_DIR, "happy"),
    os.path.join(SCRIPT_DIR, "sad"),
    os.path.join(SCRIPT_DIR, "suprised"),
]

TARGET_SIZE      = 96
BATCH_SIZE       = 32
REP_PER_CLASS    = 50   # images per folder for representative dataset

os.makedirs(GEN_DIR, exist_ok=True)

# ============================================================
# Load class names
# ============================================================
with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
    class_names = json.load(f)
num_classes = len(class_names)
print("Classes:", class_names)

# ============================================================
# Load test dataset (images arrive as [0, 255] float32)
# The model includes preprocess_input internally, so no extra
# normalization is needed here.
# ============================================================
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

# ============================================================
# Load Keras model
# Keras 3 resolves BatchNormalization directly from keras.layers,
# bypassing custom_objects. Monkey-patch __init__ to drop the
# legacy renorm params saved by Keras 2 / TF2.x.
# ============================================================
_orig_bn_init = tf.keras.layers.BatchNormalization.__init__

def _patched_bn_init(self, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    _orig_bn_init(self, **kwargs)

tf.keras.layers.BatchNormalization.__init__ = _patched_bn_init

print(f"\nLoading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

tf.keras.layers.BatchNormalization.__init__ = _orig_bn_init  # restore
model.summary()

# ============================================================
# Evaluate original Keras model
# ============================================================
print("\n--- Float32 Keras model ---")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# ============================================================
# Build representative dataset from training images
# Images are loaded raw [0, 255]; preprocess_input is inside
# the model, so representative data should also be raw.
# ============================================================
def load_representative_images(dirs, n_per_dir=REP_PER_CLASS):
    """Load up to n_per_dir images from each directory as raw [0,255] float32."""
    images = []
    for folder in dirs:
        if not os.path.isdir(folder):
            print(f"  WARNING: calibration folder not found, skipping: {folder}")
            continue
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n_per_dir]
        for fname in files:
            try:
                img = Image.open(os.path.join(folder, fname)).convert("RGB")
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                images.append(np.array(img, dtype=np.float32))  # [0, 255]
            except Exception:
                pass
        print(f"  Loaded {len(files)} images from {os.path.basename(folder)}/")
    return np.array(images, dtype=np.float32)

print("\nBuilding representative dataset from real ESP camera images...")
rep_data = load_representative_images(REALWORLD_DIRS)
print(f"Representative dataset shape: {rep_data.shape}")

# ============================================================
# Float32 TFLite conversion
# ============================================================
print("\n--- Converting to Float32 TFLite ---")
conv_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f32 = conv_f32.convert()
f32_path = os.path.join(GEN_DIR, "emotion_mobilenet_f32.tflite")
with open(f32_path, "wb") as f:
    f.write(tflite_f32)
print(f"Float32 TFLite: {len(tflite_f32) / 1024:.1f} KB  →  {f32_path}")

# ============================================================
# INT8 post-training quantization
# ============================================================
print("\n--- Converting to INT8 TFLite ---")
conv_int8 = tf.lite.TFLiteConverter.from_keras_model(model)

def representative_dataset():
    for i in range(0, len(rep_data), BATCH_SIZE):
        yield [rep_data[i:i + BATCH_SIZE]]

conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
conv_int8.representative_dataset = representative_dataset
conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv_int8.inference_input_type = tf.int8
conv_int8.inference_output_type = tf.int8

tflite_int8 = conv_int8.convert()
int8_path = os.path.join(GEN_DIR, "emotion_mobilenet_int8.tflite")
with open(int8_path, "wb") as f:
    f.write(tflite_int8)
print(f"INT8 TFLite:    {len(tflite_int8) / 1024:.1f} KB  →  {int8_path}")

# Read quantization params from INT8 model
interp_int8 = tf.lite.Interpreter(model_content=tflite_int8)
interp_int8.allocate_tensors()
inp_det = interp_int8.get_input_details()[0]
out_det = interp_int8.get_output_details()[0]
input_scale, input_zp   = inp_det['quantization']
output_scale, output_zp = out_det['quantization']
print(f"Input  quantization: scale={input_scale}, zero_point={input_zp}")
print(f"Output quantization: scale={output_scale}, zero_point={output_zp}")

# ============================================================
# Evaluate TFLite models on test set
# ============================================================
def eval_tflite(tflite_bytes, test_ds, class_names, label):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp   = inp['quantization']
    out_scale, out_zp = out['quantization']
    is_int8 = inp['dtype'] == np.int8

    y_true, y_pred = [], []
    for images, labels in test_ds:
        for img, lbl in zip(images.numpy(), labels.numpy()):
            img_f32 = img.astype(np.float32)  # [0, 255], preprocess_input is inside model
            if is_int8:
                img_q = np.clip(np.round(img_f32 / in_scale) + in_zp, -128, 127).astype(np.int8)
                interp.set_tensor(inp['index'], img_q.reshape(1, TARGET_SIZE, TARGET_SIZE, 3))
            else:
                interp.set_tensor(inp['index'], img_f32.reshape(1, TARGET_SIZE, TARGET_SIZE, 3))
            interp.invoke()
            raw_out = interp.get_tensor(out['index'])[0]
            if is_int8:
                raw_out = (raw_out.astype(np.float32) - out_zp) * out_scale
            y_pred.append(int(np.argmax(raw_out)))
            y_true.append(int(lbl))

    acc = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    print(f"\n=== {label} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return acc

print("\nEvaluating Float32 TFLite...")
eval_tflite(tflite_f32, test_ds, class_names, "Float32 TFLite")

print("\nEvaluating INT8 TFLite...")
eval_tflite(tflite_int8, test_ds, class_names, "INT8 TFLite")

# ============================================================
# Export C files for ESP32
# ============================================================
def write_model_h(path, defines, class_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write("// Generated by convert_and_export.py — do not edit manually\n\n")
        for k, v in defines.items():
            f.write(f"#define {k} {v}\n")
        f.write("\n")
        f.write("// Initializer list for class name array, in model output order\n")
        names_str = ", ".join(f'"{n}"' for n in class_names)
        f.write(f"#define CLASS_NAMES_INIT {{ {names_str} }}\n\n")
        f.write("extern const unsigned char model_binary[];\n")
        f.write("extern const unsigned int model_binary_len;\n\n")
        f.write("#endif  // MODEL_H\n")

def write_model_c(path, tflite_bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write('#include "model.h"\n\n')
        f.write("const unsigned char model_binary[] = {\n")
        for i, byte in enumerate(tflite_bytes):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write(f"\n}};\n\n")
        f.write(f"const unsigned int model_binary_len = {len(tflite_bytes)};\n")

defines = {
    "TARGET_SIZE":       TARGET_SIZE,
    "NUM_CLASSES":       num_classes,
    "INPUT_SCALE":       f"{input_scale:.10f}f",
    "INPUT_ZERO_POINT":  int(input_zp),
    "OUTPUT_SCALE":      f"{output_scale:.10f}f",
    "OUTPUT_ZERO_POINT": int(output_zp),
}

print(f"\nExporting C files...")
write_model_h(MODEL_H_PATH, defines, class_names)
write_model_c(MODEL_C_PATH, tflite_int8)

print(f"  {MODEL_H_PATH}")
print(f"  {MODEL_C_PATH}")
print("\nDone.")
