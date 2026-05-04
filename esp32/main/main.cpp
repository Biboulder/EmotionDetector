#include <cstdio>
#include <cstdint>
#include <cstring>

// ESP includes
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "driver/usb_serial_jtag.h"

// Project includes
#include "camera.h"
#include "preprocess.h"
#include "inference.h"
#include "model.h"

static const char *TAG = "EmotionDetect";

static const char *const CLASS_NAMES[NUM_CLASSES] = CLASS_NAMES_INIT;

// Protocol: viewer sends 'S' to start streaming. Each iteration the ESP sends
//   "\n===FRAME:<W>:<H>===\n"   (ASCII preamble)
//   <W * H * 2 bytes>           (raw big-endian RGB565)
//   "PRED:c0:p0,c1:p1,...\n"    (ASCII predictions, one line)
static constexpr size_t USB_CHUNK_SIZE = 256;
static constexpr TickType_t USB_TIMEOUT = pdMS_TO_TICKS(1000);

EXT_RAM_BSS_ATTR static uint8_t s_rgb565_buf[FRAME_W * FRAME_H * 2];
EXT_RAM_BSS_ATTR static int8_t  s_input_buf[TARGET_SIZE * TARGET_SIZE * 3];

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

static void usb_write_all(const uint8_t *buf, size_t len)
{
    size_t offset = 0;
    while (offset < len) {
        size_t to_write = (offset + USB_CHUNK_SIZE < len) ? USB_CHUNK_SIZE : (len - offset);
        int written = usb_serial_jtag_write_bytes(buf + offset, to_write, USB_TIMEOUT);
        if (written <= 0) {
            vTaskDelay(1);
            continue;
        }
        offset += written;
    }
}

void setup(void)
{
    init_nvs();

    if (!camera_init()) {
        ESP_LOGE(TAG, "Camera init failed - halting");
        abort();
    }
    if (!inference_init()) {
        ESP_LOGE(TAG, "Inference init failed - halting");
        abort();
    }

    // Install USB-Serial-JTAG driver for direct binary I/O.
    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = 1024,
        .rx_buffer_size = 256,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));

    ESP_LOGI(TAG, "Ready. Send 'S' to start streaming (%dx%d, %d classes).",
             FRAME_W, FRAME_H, NUM_CLASSES);

    // Wait for 'S' from the viewer.
    char c;
    do {
        int r = usb_serial_jtag_read_bytes(&c, 1, portMAX_DELAY);
        if (r < 0) abort();
    } while (c != 'S');

    // Suppress ESP_LOG output once streaming starts; log lines mixed into the
    // binary frame data would corrupt it on the viewer side.
    esp_log_level_set("*", ESP_LOG_NONE);
}

void loop(void)
{
    if (!camera_capture_frame(s_rgb565_buf)) {
        vTaskDelay(pdMS_TO_TICKS(100));
        return;
    }

    preprocess_frame(s_rgb565_buf, s_input_buf);
    if (!inference_run(s_input_buf, s_probs)) {
        vTaskDelay(pdMS_TO_TICKS(100));
        return;
    }

    // 1. Preamble (leading \n separates it from any prior content).
    char preamble[64];
    int n = snprintf(preamble, sizeof(preamble),
                     "\n===FRAME:%d:%d===\n", FRAME_W, FRAME_H);
    usb_write_all(reinterpret_cast<const uint8_t *>(preamble), (size_t)n);

    // 2. Raw RGB565 frame.
    usb_write_all(s_rgb565_buf, sizeof(s_rgb565_buf));

    // 3. Prediction line.
    char pred[256];
    int p = snprintf(pred, sizeof(pred), "PRED:");
    for (int i = 0; i < NUM_CLASSES && p < (int)sizeof(pred) - 32; i++) {
        p += snprintf(pred + p, sizeof(pred) - p, "%s%s:%.4f",
                      (i == 0) ? "" : ",", CLASS_NAMES[i], s_probs[i]);
    }
    p += snprintf(pred + p, sizeof(pred) - p, "\n");
    usb_write_all(reinterpret_cast<const uint8_t *>(pred), (size_t)p);

    // ~2 fps; the viewer can't keep up much faster over USB-CDC anyway.
    vTaskDelay(pdMS_TO_TICKS(500));
}

extern "C" void app_main(void)
{
    setup();
    while (true) {
        loop();
    }
}
