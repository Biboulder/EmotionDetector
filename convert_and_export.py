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

REP_PER_CLASS    = 340   # images per folder for INT8 calibration (~900 total)
TEST_SIZE        = 20    # images per folder for accuracy evaluation (taken after calibration slice)

# Notebook (Mobilenet_v2.ipynb) trains with this exact class ordering:
#   image_dataset_from_directory(class_names=CLASSES) → softmax index 0=surprise, 1=happy, 2=sad
EXPECTED_CLASSES = ["surprise", "happy", "sad"]

# ESP-camera folder name → model class label. Absorbs the "suprised" typo.
FOLDER_TO_CLASS = {
    "happy":    "happy",
    "sad":      "sad",
    "suprised": "surprise",
}

os.makedirs(GEN_DIR, exist_ok=True)

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
# compile=False skips restoring the optimizer/loss, which avoids errors when the
# model was saved with a custom loss_fn that isn't importable here.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

tf.keras.layers.BatchNormalization.__init__ = _orig_bn_init  # restore
model.summary()

# ============================================================
# Derive TARGET_SIZE from the model's actual input shape — must
# happen after load_model so the conversion never silently uses
# a hardcoded resolution that disagrees with the trained model.
# ============================================================
_, _H, _W, _C = model.input_shape
assert _C == 3 and _H == _W, f"Unexpected model input shape {model.input_shape}"
TARGET_SIZE = int(_H)
print(f"Detected model input: {TARGET_SIZE}x{TARGET_SIZE}x{_C}")

# ============================================================
# Class names — Mobilenet_v2.ipynb does NOT write class_names.json,
# so any existing file is from the deprecated mobileNET.py and has
# the wrong order (['happy','neutral','sad']). Trust the notebook's
# explicit class_names=CLASSES ordering instead.
# ============================================================
class_names = None
if os.path.isfile(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            class_names = json.load(f)
    except Exception:
        class_names = None

if class_names != EXPECTED_CLASSES:
    print(f"WARNING: class_names.json={class_names}; overriding with notebook order {EXPECTED_CLASSES}")
    class_names = list(EXPECTED_CLASSES)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f)

num_classes = len(class_names)
print("Classes (model output order):", class_names)

# ============================================================
# Build representative dataset from training images
# Images are loaded raw [0, 255]; preprocess_input is inside
# the model, so representative data should also be raw.
# ============================================================
def load_realworld_split(dirs, n_calib=REP_PER_CLASS, n_test=TEST_SIZE):
    """Load images from each directory, split into calibration and test slices.

    Test labels are mapped to the *model's* softmax index via FOLDER_TO_CLASS
    + class_names, so the "suprised" folder typo and any folder/output-order
    differences resolve to the correct ground-truth class.

    Returns:
        calib_images: np.array [0,255] float32, shape (N_calib, H, W, 3)
        test_images:  np.array [0,255] float32, shape (N_test,  H, W, 3)
        test_labels:  list of int, model softmax indices
    """
    calib_images, test_images, test_labels = [], [], []
    for folder in dirs:
        base = os.path.basename(folder)
        if not os.path.isdir(folder):
            print(f"  WARNING: folder not found, skipping: {folder}")
            continue
        if base not in FOLDER_TO_CLASS:
            print(f"  WARNING: no class mapping for folder {base!r}, skipping")
            continue
        cls_name = FOLDER_TO_CLASS[base]
        cls_idx  = class_names.index(cls_name)

        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
        calib_files = files[:n_calib]
        test_files  = files[n_calib : n_calib + n_test]
        for fname in calib_files:
            try:
                img = Image.open(os.path.join(folder, fname)).convert("RGB")
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                calib_images.append(np.array(img, dtype=np.float32))
            except Exception:
                pass
        for fname in test_files:
            try:
                img = Image.open(os.path.join(folder, fname)).convert("RGB")
                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                test_images.append(np.array(img, dtype=np.float32))
                test_labels.append(cls_idx)
            except Exception:
                pass
        print(f"  {base}/ → {cls_name} (idx {cls_idx}): "
              f"{len(calib_files)} calib, {len(test_files)} test")
    return (np.array(calib_images, dtype=np.float32),
            np.array(test_images,  dtype=np.float32),
            test_labels)

print("\nLoading real ESP camera images (calibration + test split)...")
rep_data, realworld_test_images, realworld_test_labels = load_realworld_split(REALWORLD_DIRS)
print(f"Calibration set: {rep_data.shape}")
print(f"Real-world test set: {realworld_test_images.shape}")

# ============================================================
# Calibration diagnostic — surface activation ranges that drive
# the INT8 scale/zero-point per the lecture: s_a and z_a are
# computed from r_min/r_max observed during calibration. If a
# layer's max ≫ p99, outliers are inflating the scale.
# ============================================================
print(f"\nCalib pixel stats: min={rep_data.min():.1f} "
      f"max={rep_data.max():.1f} mean={rep_data.mean():.1f}")
try:
    relu_layers = [l for l in model.layers if 'relu' in l.name.lower()][:6]
    if relu_layers:
        probe = tf.keras.Model(model.input, [l.output for l in relu_layers])
        acts  = probe(rep_data[:64], training=False)
        if not isinstance(acts, list):
            acts = [acts]
        for l, a in zip(relu_layers, acts):
            a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
            print(f"  {l.name:35s} max={a.max():7.2f}  p99={np.percentile(a, 99):7.2f}")
except Exception as e:
    print(f"  (activation probe skipped: {e})")

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
    # TFLiteConverter expects each yield to be a single example shape
    # [1, H, W, 3]. Yielding batches caused TF to use the batch as one
    # "sample" for calibration, so r_min/r_max were derived from
    # batch-wide statistics instead of per-image extremes.
    for i in range(len(rep_data)):
        yield [rep_data[i:i + 1].astype(np.float32)]

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
# Evaluate TFLite models
# Supports two input forms:
#   - tf.data dataset (images [0,255] float32, int labels)
#   - (images_array, labels_list) from the real-world split
# ============================================================
def eval_tflite(tflite_bytes, test_source, class_names, label):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp   = inp['quantization']
    out_scale, out_zp = out['quantization']
    is_int8 = inp['dtype'] == np.int8

    y_true, y_pred = [], []

    if isinstance(test_source, tuple):
        # Real-world split: (np.array of images, list of int labels)
        images_arr, labels_list = test_source
        for img_f32, lbl in zip(images_arr, labels_list):
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
    else:
        # tf.data dataset
        for images, labels in test_source:
            for img, lbl in zip(images.numpy(), labels.numpy()):
                img_f32 = img.astype(np.float32)
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

# test labels are now model softmax indices, so target_names must be
# the model's class_names (model output order), not folder names.
realworld_test = (realworld_test_images, realworld_test_labels)

print("\n--- Evaluating Float32 TFLite on real-world ESP camera images ---")
f32_acc = eval_tflite(tflite_f32, realworld_test, class_names, "Float32 TFLite (real-world)")

print("\n--- Evaluating INT8 TFLite on real-world ESP camera images ---")
int8_acc = eval_tflite(tflite_int8, realworld_test, class_names, "INT8 TFLite (real-world)")

print(f"\nSummary on ESP-camera real-world set:")
print(f"  F32  accuracy: {f32_acc:.4f}")
print(f"  INT8 accuracy: {int8_acc:.4f}  (Δ vs F32 = {int8_acc - f32_acc:+.4f})")

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
