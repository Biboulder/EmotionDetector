#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load model and allocate tensor arena. Must be called once before inference_run().
bool inference_init(void);

// Run one inference pass.
//   input_int8   – quantised input tensor, TARGET_SIZE * TARGET_SIZE * 3 int8 values
//   output_probs – dequantised softmax probabilities, NUM_CLASSES float values
// Returns false if Invoke() fails.
bool inference_run(const int8_t *input_int8, float *output_probs);

#ifdef __cplusplus
}
#endif
