#pragma once

#include <stdint.h>
#include "camera.h"
#include "model.h"

// Convert a raw 96×96 RGB565 camera frame to an INT8 tensor ready for the model.
// Camera is configured at the model's native resolution, so this only unpacks
// RGB565 → RGB888 and quantises each channel — no crop, no resize.
//
// output_int8 must point to a buffer of at least TARGET_SIZE * TARGET_SIZE * 3 bytes.
void preprocess_frame(const uint8_t *rgb565_frame, int8_t *output_int8);
