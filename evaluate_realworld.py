"""
Step 12: Evaluate quantized TFLite model on real-world images.

Expects images organized in class-named subfolders, e.g.:
  <images_root>/
    happy/   image1.png  image2.png  ...
    neutral/ image1.png  ...
    sad/     image1.png  ...

Applies OpenCV Haar cascade face detection to crop faces before
inference. Falls back to center crop if no face is detected.

Usage:
  python evaluate_realworld.py --images <folder>
  python evaluate_realworld.py --images .        # defaults to current dir
  python evaluate_realworld.py --images . --no-face-detect
"""

import os
import json
import argparse
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TARGET_SIZE = 160
FACE_PAD    = 0.30   # fractional padding added around detected face bounding box


def center_crop_square(img_bgr):
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_bgr[y0:y0 + side, x0:x0 + side]


def detect_and_crop_face(img_bgr):
    """Return a square crop around the largest detected face, or center crop."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return center_crop_square(img_bgr), False

    # Largest face by area
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

    # Add padding around detected face
    pad = int(max(fw, fh) * FACE_PAD)
    h, w = img_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + fw + pad)
    y2 = min(h, y + fh + pad)
    crop = img_bgr[y1:y2, x1:x2]

    # Pad to square if necessary
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    sq = np.zeros((side, side, 3), dtype=np.uint8)
    sq[(side - ch) // 2:(side - ch) // 2 + ch,
       (side - cw) // 2:(side - cw) // 2 + cw] = crop
    return sq, True


def prepare_image(img_bgr, use_face_detect=True):
    """Crop, resize to TARGET_SIZE, return float32 in [0, 255] (model preprocess_input is internal)."""
    if use_face_detect:
        crop, found = detect_and_crop_face(img_bgr)
        if not found:
            print("    [no face detected, using center crop]")
    else:
        crop = center_crop_square(img_bgr)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized  = np.array(Image.fromarray(crop_rgb).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS),
                        dtype=np.float32)  # [0, 255]
    return resized


def load_interpreter(tflite_path):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return interp, inp, out


def run_inference(interp, inp, out, img_f32):
    in_scale, in_zp   = inp['quantization']
    out_scale, out_zp = out['quantization']
    is_int8 = inp['dtype'] == np.int8

    if is_int8:
        img_q = np.clip(np.round(img_f32 / in_scale) + in_zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp['index'], img_q.reshape(1, TARGET_SIZE, TARGET_SIZE, 3))
    else:
        interp.set_tensor(inp['index'], img_f32.reshape(1, TARGET_SIZE, TARGET_SIZE, 3))

    interp.invoke()
    raw = interp.get_tensor(out['index'])[0]

    if is_int8:
        raw = (raw.astype(np.float32) - out_zp) * out_scale
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  default=".",
                        help="Root folder with emotion-class subfolders (default: current dir)")
    parser.add_argument("--model",   default=os.path.join(SCRIPT_DIR, "generated_mobilenet",
                                                           "emotion_mobilenet_int8.tflite"),
                        help="Path to TFLite model")
    parser.add_argument("--classes", default=os.path.join(SCRIPT_DIR, "generated_mobilenet",
                                                           "class_names.json"),
                        help="Path to class_names.json")
    parser.add_argument("--no-face-detect", action="store_true",
                        help="Skip face detection; use center square crop instead")
    args = parser.parse_args()

    with open(args.classes) as f:
        class_names = json.load(f)
    print(f"Model classes: {class_names}")
    print(f"Model:         {args.model}")
    print(f"Images root:   {args.images}")
    print(f"Face detect:   {not args.no_face_detect}\n")

    interp, inp_det, out_det = load_interpreter(args.model)

    y_true, y_pred = [], []

    for cls_folder in sorted(os.listdir(args.images)):
        cls_dir = os.path.join(args.images, cls_folder)
        if not os.path.isdir(cls_dir):
            continue

        # Match folder name case-insensitively against known classes
        cls_lower = cls_folder.lower()
        if cls_lower not in class_names:
            print(f"Skipping '{cls_folder}' — not in model classes {class_names}")
            continue
        cls_idx = class_names.index(cls_lower)

        img_files = [f for f in sorted(os.listdir(cls_dir))
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        print(f"[{cls_folder}]  {len(img_files)} images")

        for fname in img_files:
            fpath = os.path.join(cls_dir, fname)
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                print(f"  Could not load: {fname}")
                continue

            img_f32  = prepare_image(img_bgr, use_face_detect=not args.no_face_detect)
            probs    = run_inference(interp, inp_det, out_det, img_f32)
            pred_idx = int(np.argmax(probs))
            conf     = float(probs[pred_idx])

            y_true.append(cls_idx)
            y_pred.append(pred_idx)
            marker = "OK" if pred_idx == cls_idx else "WRONG"
            print(f"  [{marker}] {fname}  pred={class_names[pred_idx]} ({conf:.2f})")

    if not y_true:
        print("\nNo labeled images found. Make sure subfolders match class names:", class_names)
        return

    acc = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    print(f"\n{'='*50}")
    print(f"Real-world evaluation  ({len(y_true)} images)")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
