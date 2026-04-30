#pragma once

#include <stdbool.h>
#include <stdint.h>

// Camera output resolution: square 96x96 RGB565 — matches model input directly.
// Capturing at the model's native size skips center-crop and resize entirely.
#define FRAME_W 96
#define FRAME_H 96

#ifdef __cplusplus
extern "C" {
#endif

bool camera_init(void);

// Fills rgb565_buffer with FRAME_W * FRAME_H * 2 bytes of raw RGB565 data.
bool camera_capture_frame(uint8_t *rgb565_buffer);

#ifdef __cplusplus
}
#endif
