# EmotionDetector

Facial emotion recognition on XIAO ESP32-S3 Sense. A MobileNetV2 α=0.5 model is trained on AffectNet, quantized to INT8, and deployed as a TFLite Micro application that runs inference from the camera in real time.

**Classes:** happy, neutral, sad  
**Input:** 96×96 RGB — camera captures at model resolution directly, no crop or resize  
**Model size:** ~1.3 MB TFLite INT8

---

## VS Code Setup

### 1. Add ESP-IDF paths to VS Code User Settings

Open **Ctrl+Shift+P → Preferences: Open User Settings (JSON)** and add inside the outer `{}`:

```json
"idf.espIdfPathWin": "C:/Users/<your-user-name>/esp/esp-idf",
"idf.toolsPathWin": "C:/Users/<your-user-name>/.espressif"
```

These are referenced by `.vscode/c_cpp_properties.json` for IntelliSense and by the ESP-IDF terminal profile below.

### 2. Python IntelliSense

This project reuses the shared venv from `DTU-02214` (already configured in `.vscode/settings.json`). If Pylance shows import errors, press **Ctrl+Shift+P → Python: Select Interpreter** and choose the `.venv` from `DTU-02214`.

### 3. C/C++ IntelliSense

`.vscode/c_cpp_properties.json` is pre-configured for ESP-IDF v5.5 on Windows. IntelliSense will fully resolve after a successful `idf.py build` (which generates `esp32/build/compile_commands.json`).

### 4. ESP-IDF terminal

An **ESP-IDF (cmd)** terminal profile is defined in `.vscode/settings.json`. Open it via the terminal dropdown (▾ next to the `+` button) — it auto-sources `export.bat` so `idf.py` is available immediately. Use this terminal for all ESP32 build/flash commands.

---

## Workflow

### Step 1 — Install Python dependencies

This project shares the virtual environment from `DTU-02214`. If packages are missing:

```bash
# activate the shared venv first
pip install -r requirements.txt
```

Requires a Kaggle API key at `~/.kaggle/kaggle.json` for dataset download.

### Step 2 — Preprocess dataset

```bash
jupyter notebook data_preprocessing.ipynb
```

Downloads AffectNet from Kaggle, filters by size, splits into train/val/test, applies augmentation, and writes resized images to `emotion_dataset_mobilenet/`.

### Step 3 — Train model

```bash
python mobileNET.py
```

Two-phase training: frozen MobileNetV2 base → fine-tune upper layers. Key outputs in `generated_mobilenet/`:

- `emotion_mobilenet_finetuned.keras` — best checkpoint
- `class_names.json` — label ordering used by all downstream steps
- `inspect_misclassified/` — misclassified test samples grouped by true/predicted pair

### Step 4 — Convert to TFLite and export C files

```bash
python convert_and_export.py <model.keras>
```

Pass the `.keras` file to convert (relative paths are resolved from the project root). Example:

```bash
python convert_and_export.py best_model_uploaded.keras
```

Converts to Float32 then INT8 (post-training quantization). INT8 calibration uses real images captured by the ESP camera (`happy/`, `sad/`, `suprised/`), which better matches the sensor's actual colour distribution. Writes:

- `generated_mobilenet/emotion_mobilenet_f32.tflite`
- `generated_mobilenet/emotion_mobilenet_int8.tflite`
- `esp32/main/model.h` — quantization params + class names
- `esp32/main/model.c` — INT8 model binary as a C array

**Run this before building the ESP32 firmware.**

### Step 5 — Evaluate on real-world images (optional)

Organize captured images into subfolders named after the emotion classes:

```
happy/   img1.jpg  img2.jpg  ...
neutral/ img1.jpg  ...
sad/     img1.jpg  ...
```

```bash
# with OpenCV face detection (default)
python evaluate_realworld.py --images .

# center-crop only, no face detection
python evaluate_realworld.py --images . --no-face-detect
```

Falls back to center-crop when no face is detected. Prints per-class accuracy, classification report, and confusion matrix.

### Step 6 — Build and flash ESP32

Open the **ESP-IDF (cmd)** terminal (see VS Code Setup above), then:

```bash
cd esp32
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

The first build downloads managed components (`espressif/esp-tflite-micro`, `espressif/esp32-camera`) — takes several minutes. The serial monitor prints per-frame emotion predictions with latency:

```
Predicted: happy (87.3%)  [12 ms]
```

---

## Configuration

All training hyperparameters are at the top of `mobileNET.py`:

| Parameter | Default | Notes |
|---|---|---|
| `TARGET_SIZE` | 160 | Input image size (px) |
| `IMAGES_PER_CLASS` | 3000 | Dataset cap per class |
| `BATCH_SIZE` | 32 | |
| `HEAD_EPOCHS` | 12 | Phase 1 (frozen base) |
| `FINE_TUNE_EPOCHS` | 8 | Phase 2 (upper layers) |
| `ALPHA` | 0.5 | MobileNetV2 width (0.35 = smaller, 0.75 = larger) |
| `FINE_TUNE_AT` | 125 | Freeze MobileNetV2 layers below this index |

To train on different emotions, change `label_map` in `mobileNET.py` (keys are AffectNet folder IDs: 0=neutral, 1=happy, 2=sad, 3=surprise, 4=fear, 5=disgust, 6=anger).

---

## Troubleshooting

**`AllocateTensors() failed`** — tensor arena too small. Increase `TENSOR_ARENA_SIZE` in `esp32/main/inference.cpp` (currently 1.5 MB). The log line `Arena used: N bytes` shows the actual requirement.

**Out of memory during training** — reduce `BATCH_SIZE` or `IMAGES_PER_CLASS`.

**Poor accuracy after quantization** — increase `REP_PER_CLASS` in `convert_and_export.py` to give the quantizer more calibration data.

**Camera init failed on ESP32** — confirm target is `esp32s3` and that the board is the XIAO ESP32-S3 Sense (pin assignments in `camera.cpp` are specific to that board).

**`<model>.keras` not found** — run `mobileNET.py` first to generate it; `.keras` files are gitignored.

**`idf.py` not found** — use the **ESP-IDF (cmd)** terminal profile, not the default PowerShell terminal.
