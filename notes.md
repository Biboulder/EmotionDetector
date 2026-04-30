Yes, the inference is taking approximately 11.7 seconds per frame. This is the critical problem.

To break down the timing:

Preprocessing (center crop, resize, quantize): 23 ms — normal, expected
Inference (TFLite Micro convolution operations): 11,736 ms — this is extremely slow
For context, MobileNetV2 inference on ESP32-S3 should typically complete in 1–3 seconds, not 11 seconds. This 11.7-second time is why the task watchdog (5-second timeout) was triggering—the inference couldn't finish before the device reset.

What This Tells Us
The quantized INT8 model is either:

Poorly quantized — the representative dataset used in convert_and_export.py didn't capture the distribution of weights/activations well, so the INT8 version has degraded accuracy and is spending extra cycles trying to compute
Computationally too heavy for the ESP32-S3 clock speed — even at α=0.5, the model is still too large
Next Step (Your Priority: "1+4, then 2")
You should now re-run convert_and_export.py to:

Check the accuracy comparison: Float32 model vs. INT8 quantized model on your test set
If INT8 accuracy dropped significantly (e.g., >5% drop), the representative dataset was poor
Regenerate the INT8 model with a better/larger representative dataset
Alternatively, try reducing model size: change ALPHA = 0.5 → ALPHA = 0.35 in mobileNET.py, retrain, then re-quantize
The 11.7-second inference time is not acceptable. It needs to be reduced to <2–3 seconds before deployment.


-----------------------------------------
Why Build a Representative Dataset?
The representative dataset is critical for INT8 quantization quality—it's not just for testing.

When you convert a Float32 model to INT8 (full integer quantization), you're mapping floating-point weights and activations to 8-bit integer ranges. TensorFlow needs to know:

What range of values appear in the network when real data flows through it
How to best scale those ranges to fit into [-128, 127]
The representative dataset (300 images per class from the training set) is used to collect statistics about these activations so TensorFlow can compute optimal scale and zero_point values for each layer.

If the representative dataset is poor, the quantization mapping will be bad → accuracy drops → inference may need more cycles to compensate → slow inference (which is your 11.7-second problem).

Why Two TFLite Files?
emotion_mobilenet_f32.tflite (Float32):

Reference baseline for comparing accuracy
Used for testing on desktop/laptop
Tells us if INT8 quantization degraded accuracy
emotion_mobilenet_int8.tflite (INT8 quantized):

Deployed to ESP32
~4× smaller file size
Should be faster, but only if quantization is done well
Your 11.7-Second Problem
The slow inference strongly suggests poor quantization. When you re-run convert_and_export.py, you'll see:

Float32 accuracy (baseline)
INT8 accuracy (quantized version)
Quantization parameters (INPUT_SCALE, OUTPUT_SCALE, etc.)
If INT8 accuracy is significantly worse than F32, that's your problem: the representative dataset (300 images) isn't representative enough. Solution: increase REP_PER_CLASS from 300 → 500 or 1000 and re-run the script.


The load_representative_images function does not classify images at all — it has no knowledge of which emotion a file belongs to. Here's what it actually does:

Purpose: It provides a calibration dataset for INT8 quantization, not for training or evaluation. The quantizer uses these images purely to observe the range of activation values flowing through the network, so it can pick good scale/zero-point parameters. The actual class labels are irrelevant for this purpose.

How it works (convert_and_export.py:75-91):

It iterates over subdirectories of emotion_dataset/train/ — whatever folders exist there (e.g. happy/, sad/, neutral/)
It picks up to REP_PER_CLASS=300 non-augmented image files from each folder
It loads each image as raw [0, 255] float32 pixels
Returns a flat array of all images — no labels are attached