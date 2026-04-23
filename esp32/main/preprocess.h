#pragma once

#include <stdint.h>
#include "camera.h"
#include "model.h"

// Convert a raw RGB565 camera frame to an INT8 tensor ready for model input.
//
// Pipeline:
//   1. Center-crop FRAME_W×FRAME_H (320×240) to a FRAME_H×FRAME_H (240×240) square.
//   2. Nearest-neighbour resize to TARGET_SIZE×TARGET_SIZE (96×96).
//   3. Quantize each channel: int8 = clamp(round(pixel / INPUT_SCALE) + INPUT_ZERO_POINT)
//      where pixel ∈ [0, 255] (the model includes preprocess_input internally).
//
// output_int8 must point to a buffer of at least TARGET_SIZE * TARGET_SIZE * 3 bytes.
void preprocess_frame(const uint8_t *rgb565_frame, int8_t *output_int8);
