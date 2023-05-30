#include "blitz/array.h"
void compute_fft_2D_R2C(blitz::Array<float, 2> &grid, void *gpu_slab);
void *allocate_cuda_slab(size_t nGrid);