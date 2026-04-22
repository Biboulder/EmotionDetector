#include "inference.h"

#include "esp_log.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

static const char *TAG = "Inference";

// Tensor arena size — MobileNetV2 α=0.5 at 160×160 needs ~1–1.5 MB.
// Allocated from PSRAM (XIAO ESP32-S3 Sense has 8 MB PSRAM).
// Tune downward in 64 KB steps until AllocateTensors() fails, then add 128 KB back.
#define TENSOR_ARENA_SIZE (1536 * 1024)  // 1.5 MB initial estimate

static const tflite::Model   *s_model      = nullptr;
static tflite::MicroInterpreter *s_interp  = nullptr;
static uint8_t               *s_arena      = nullptr;
static TfLiteTensor          *s_input      = nullptr;
static TfLiteTensor          *s_output     = nullptr;

bool inference_init(void)
{
    // Allocate tensor arena in PSRAM
    s_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE,
                                          MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_arena) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena (%d bytes) in PSRAM", TENSOR_ARENA_SIZE);
        return false;
    }
    ESP_LOGI(TAG, "Tensor arena: %d KB in PSRAM", TENSOR_ARENA_SIZE / 1024);

    // Load TFLite flatbuffer
    s_model = tflite::GetModel(model_binary);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch: got %d, expected %d",
                 s_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Register ops used by MobileNetV2 INT8
    static tflite::MicroMutableOpResolver<13> resolver;
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMean();           // GlobalAveragePooling2D → ReduceMean
    resolver.AddReshape();
    resolver.AddFullyConnected(); // Dense layers
    resolver.AddSoftmax();
    resolver.AddPad();
    resolver.AddPadV2();
    resolver.AddRelu6();          // sometimes fused, sometimes explicit
    resolver.AddMaxPool2D();

    static tflite::MicroInterpreter static_interp(
        s_model, resolver, s_arena, TENSOR_ARENA_SIZE);
    s_interp = &static_interp;

    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed — increase TENSOR_ARENA_SIZE");
        return false;
    }

    s_input  = s_interp->input(0);
    s_output = s_interp->output(0);

    ESP_LOGI(TAG, "Input  type=%s shape=[%d,%d,%d,%d]",
             TfLiteTypeGetName(s_input->type),
             s_input->dims->data[0], s_input->dims->data[1],
             s_input->dims->data[2], s_input->dims->data[3]);
    ESP_LOGI(TAG, "Output type=%s shape=[%d,%d]",
             TfLiteTypeGetName(s_output->type),
             s_output->dims->data[0], s_output->dims->data[1]);
    ESP_LOGI(TAG, "Arena used: %u bytes", (unsigned)s_interp->arena_used_bytes());
    return true;
}

bool inference_run(const int8_t *input_int8, float *output_probs)
{
    // Copy preprocessed INT8 tensor into model input buffer
    memcpy(s_input->data.int8, input_int8, TARGET_SIZE * TARGET_SIZE * 3);

    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return false;
    }

    // Dequantise INT8 output to float probabilities
    const float out_scale = OUTPUT_SCALE;
    const int   out_zp    = OUTPUT_ZERO_POINT;
    for (int i = 0; i < NUM_CLASSES; i++) {
        output_probs[i] = ((float)s_output->data.int8[i] - out_zp) * out_scale;
    }
    return true;
}
