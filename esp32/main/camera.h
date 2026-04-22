#pragma once

#include <stdbool.h>
#include <stdint.h>

// Camera output resolution (QVGA, RGB565)
#define FRAME_W 320
#define FRAME_H 240

#ifdef __cplusplus
extern "C" {
#endif

bool camera_init(void);

// Fills rgb565_buffer with FRAME_W * FRAME_H * 2 bytes of raw RGB565 data.
bool camera_capture_frame(uint8_t *rgb565_buffer);

#ifdef __cplusplus
}
#endif
