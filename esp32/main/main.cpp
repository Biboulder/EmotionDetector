#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_attr.h"   // EXT_RAM_BSS_ATTR
#include "esp_heap_caps.h"
#include "esp_task_wdt.h" // For disabling Task Watchdog

#include "camera.h"
#include "preprocess.h"
#include "inference.h"
#include "model.h"

static const char *TAG = "EmotionDetect";

// Class names — must match order in class_names.json (alphabetical from Keras)
static const char *const CLASS_NAMES[NUM_CLASSES] = CLASS_NAMES_INIT;

// Minimum softmax probability to report a confident prediction
#define CONFIDENCE_THRESHOLD 0.60f

// Camera frame buffer: 96×96×2 bytes ≈ 18 KB  → place in PSRAM
EXT_RAM_BSS_ATTR static uint8_t  s_rgb565_buf[FRAME_W * FRAME_H * 2];

// Quantised input tensor: 96×96×3 bytes ≈ 27 KB  → place in PSRAM
EXT_RAM_BSS_ATTR static int8_t   s_input_buf[TARGET_SIZE * TARGET_SIZE * 3];

static float s_probs[NUM_CLASSES];

static void init_nvs(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

void setup(void)
{
    init_nvs();

    // Disable task watchdog to allow long inference times
    esp_task_wdt_deinit();

    if (!camera_init()) {
        ESP_LOGE(TAG, "Camera init failed — halting");
        abort();
    }

    if (!inference_init()) {
        ESP_LOGE(TAG, "Inference init failed — halting");
        abort();
    }

    ESP_LOGI(TAG, "Emotion detector ready. Confidence threshold: %.0f%%",
             CONFIDENCE_THRESHOLD * 100.0f);
}

void loop(void)
{
    // 1. Capture raw RGB565 frame
    if (!camera_capture_frame(s_rgb565_buf)) {
        ESP_LOGW(TAG, "Frame capture failed");
        vTaskDelay(pdMS_TO_TICKS(100));
        return;
    }

    // 2. Preprocess: unpack RGB565 → RGB888 and quantise to INT8 (no crop/resize — camera already at 96×96)
    int64_t t0 = esp_timer_get_time();
    preprocess_frame(s_rgb565_buf, s_input_buf);
    int64_t us_preprocess = esp_timer_get_time() - t0;

    // 3. Run TFLite Micro inference
    int64_t t1 = esp_timer_get_time();
    if (!inference_run(s_input_buf, s_probs)) {
        ESP_LOGE(TAG, "Inference failed");
        return;
    }
    int64_t us_inference = esp_timer_get_time() - t1;

    // 4. Find argmax
    int best = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (s_probs[i] > s_probs[best]) best = i;
    }

    // 5. Output results to the monitor
    ESP_LOGI(TAG, "Prediction: %s (%.1f%%) | Preprocess: %lld ms, Inference: %lld ms",
             CLASS_NAMES[best], s_probs[best] * 100.0f,
             us_preprocess / 1000, us_inference / 1000);

    for (int i = 0; i < NUM_CLASSES; i++) {
        ESP_LOGI(TAG, "  - %s: %.1f%%", CLASS_NAMES[i], s_probs[i] * 100.0f);
    }

    // Delay between inferences (e.g., 0.5 seconds), making it analyze less frequently
    vTaskDelay(pdMS_TO_TICKS(500));
}

extern "C" void app_main(void)
{
    setup();
    while (true) {
        loop();
    }
}
