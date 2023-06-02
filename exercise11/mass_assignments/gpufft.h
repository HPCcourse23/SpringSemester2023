#include "cuda_runtime.h"
#include "cufft.h"
#include <blitz/array.h>
struct stream_info
{
    void *slab;          // Pointer to 2D slab on the GPU
    void *work;          // Pointer to the work area for this stream
    cudaStream_t stream; // Stream to use to execute the FFT
};
cufftHandle make_plan(int nGrid, int n_streams, stream_info info[]);
void compute_fft_2D_R2C(blitz::Array<float, 2> &grid, stream_info *info, cufftHandle plan);