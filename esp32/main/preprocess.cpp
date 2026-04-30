#include "preprocess.h"
#include <cmath>

static_assert(FRAME_W == TARGET_SIZE && FRAME_H == TARGET_SIZE,
              "Camera frame must match model input size — no resize is performed.");

static inline int8_t quantize(uint8_t val)
{
    float q = roundf((float)val / INPUT_SCALE) + (float)INPUT_ZERO_POINT;
    if (q > 127.0f)  return  127;
    if (q < -128.0f) return -128;
    return (int8_t)q;
}

void preprocess_frame(const uint8_t *rgb565_frame, int8_t *output_int8)
{
    const int n = TARGET_SIZE * TARGET_SIZE;
    for (int i = 0; i < n; i++) {
        // RGB565 is stored big-endian by esp32-camera (high byte first)
        uint16_t pixel = ((uint16_t)rgb565_frame[i * 2] << 8)
                       |  (uint16_t)rgb565_frame[i * 2 + 1];

        uint8_t r = ((pixel >> 11) & 0x1F) << 3;
        uint8_t g = ((pixel >>  5) & 0x3F) << 2;
        uint8_t b =  (pixel        & 0x1F) << 3;

        output_int8[i * 3 + 0] = quantize(r);
        output_int8[i * 3 + 1] = quantize(g);
        output_int8[i * 3 + 2] = quantize(b);
    }
}
