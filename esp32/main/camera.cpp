#include <string.h>
#include "esp_camera.h"
#include "esp_log.h"
#include "camera.h"

static const char *TAG = "Camera";

// Sensor captures at this native square resolution; camera_capture_frame()
// center-crops the result down to FRAME_W × FRAME_H.
// 240×240 covers all typical model sizes (96, 128, 160) without a resize step.
static constexpr size_t SENSOR_FRAME_W = 240;
static constexpr size_t SENSOR_FRAME_H = 240;
static constexpr size_t CROP_X_OFFSET  = (SENSOR_FRAME_W - FRAME_W) / 2;
static constexpr size_t CROP_Y_OFFSET  = (SENSOR_FRAME_H - FRAME_H) / 2;

static_assert(FRAME_W <= SENSOR_FRAME_W && FRAME_H <= SENSOR_FRAME_H,
              "FRAME_W/H must not exceed SENSOR_FRAME_W/H (240).");

static camera_config_t make_camera_config()
{
    camera_config_t cfg = {};

    // LEDC clock for XCLK
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;

    // Data pins — XIAO ESP32-S3 Sense pinout
    cfg.pin_d0 = 15;
    cfg.pin_d1 = 17;
    cfg.pin_d2 = 18;
    cfg.pin_d3 = 16;
    cfg.pin_d4 = 14;
    cfg.pin_d5 = 12;
    cfg.pin_d6 = 11;
    cfg.pin_d7 = 48;

    // Control pins
    cfg.pin_xclk  = 10;
    cfg.pin_pclk  = 13;
    cfg.pin_vsync = 38;
    cfg.pin_href  = 47;

    // SCCB (I2C for sensor config)
    cfg.pin_sccb_sda = 40;
    cfg.pin_sccb_scl = 39;

    // Power/reset not wired on XIAO
    cfg.pin_pwdn  = -1;
    cfg.pin_reset = -1;

    cfg.xclk_freq_hz = 16000000;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.fb_count     = 2;   // multiple buffers needed for CAMERA_GRAB_LATEST
    cfg.grab_mode    = CAMERA_GRAB_LATEST;

    cfg.pixel_format = PIXFORMAT_RGB565;
    cfg.frame_size   = FRAMESIZE_240X240;  // capture square, center-crop to FRAME_W×FRAME_H
    cfg.jpeg_quality = 12;  // unused for RGB565, set to sane default

    return cfg;
}

bool camera_init(void)
{
    ESP_LOGI(TAG, "Initializing camera...");
    camera_config_t cfg = make_camera_config();
    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return false;
    }
    ESP_LOGI(TAG, "Camera ready: sensor %dx%d → crop %dx%d RGB565",
             (int)SENSOR_FRAME_W, (int)SENSOR_FRAME_H, FRAME_W, FRAME_H);
    return true;
}

bool camera_capture_frame(uint8_t *rgb565_buffer)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Failed to capture frame");
        return false;
    }

    if (fb->width != SENSOR_FRAME_W || fb->height != SENSOR_FRAME_H || fb->format != PIXFORMAT_RGB565) {
        ESP_LOGE(TAG, "Unexpected frame: %dx%d fmt=%d", fb->width, fb->height, fb->format);
        esp_camera_fb_return(fb);
        return false;
    }

    // Center-crop SENSOR_FRAME_W×SENSOR_FRAME_H down to FRAME_W×FRAME_H (2 bytes per pixel).
    for (size_t row = 0; row < FRAME_H; ++row) {
        const size_t src = ((row + CROP_Y_OFFSET) * SENSOR_FRAME_W + CROP_X_OFFSET) * 2;
        const size_t dst = row * FRAME_W * 2;
        memcpy(rgb565_buffer + dst, fb->buf + src, FRAME_W * 2);
    }

    esp_camera_fb_return(fb);
    return true;
}
