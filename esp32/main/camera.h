#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "model.h"

// Output resolution after center-crop, derived automatically from the
// generated model.h so that changing the model size only requires re-running
// convert_and_export.py (and retraining). Sensor captures at 240×240;
// camera.cpp center-crops to FRAME_W×FRAME_H.
#define FRAME_W TARGET_SIZE
#define FRAME_H TARGET_SIZE

#ifdef __cplusplus
extern "C" {
#endif

bool camera_init(void);

// Fills rgb565_buffer with FRAME_W * FRAME_H * 2 bytes of raw RGB565 data.
bool camera_capture_frame(uint8_t *rgb565_buffer);

#ifdef __cplusplus
}
#endif
