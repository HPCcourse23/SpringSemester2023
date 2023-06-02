#include "gpufft.h"
#include "cuda.h"
#include "cufft.h"

static const char *cufftGetErrorString(cufftResult cufft_error_type)
{

    switch (cufft_error_type)
    {

    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS: The CUFFT operation was performed";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN: The CUFFT plan to execute is invalid";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED: The allocation of data for CUFFT in memory failed";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE: The data type used by CUFFT is invalid";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE: The data value used by CUFFT is invalid";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR: An internal error occurred in CUFFT";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED: The execution of a plan by CUFFT failed";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED: The setup of CUFFT failed";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE: The size of the data to be used by CUFFT is invalid";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA: The data to be used by CUFFT is unaligned in memory";
    }

    return "Unknown CUFFT Error";
}

// Create a plan to do a 2D transform for the given grid (in-place)
cufftHandle make_plan(int nGrid, int n_streams, stream_info info[])
{
    int n[] = {nGrid, nGrid}; // 2D FFT of length NxN
    int inembed[] = {nGrid, 2 * (nGrid / 2 + 1)};
    int onembed[] = {nGrid, nGrid / 2 + 1};
    int howmany = 1;
    int odist = onembed[0] * onembed[1]; // Output distance is in "complex"
    int idist = 2 * odist;               // Input distance is in "real"
    int istride = 1;                     // Elements of each FFT are adjacent
    int ostride = 1;
    size_t workSize;

    cufftHandle plan;
    cufftCreate(&plan);
    cufftSetAutoAllocation(plan, 0);
    cufftMakePlanMany(plan, sizeof(n) / sizeof(n[0]), n,
                      inembed, istride, idist,
                      onembed, ostride, odist,
                      CUFFT_R2C, howmany, &workSize);
    for (auto i = 0; i < n_streams; ++i)
    {
        auto slab_size = sizeof(cufftComplex) * onembed[0] * onembed[1];
        cudaStreamCreate(&info[i].stream);
        cudaMallocAsync(&info[i].slab, slab_size, info[i].stream);
        cudaMallocAsync(&info[i].work, workSize, info[i].stream);
    }

    return plan;
}

void compute_fft_2D_R2C(blitz::Array<float, 2> &grid, stream_info *info, cufftHandle plan)
{

    auto data_size = sizeof(cufftComplex) * grid.rows() * (grid.cols() / 2 + 1);
    cudaMemcpyAsync(info->slab, grid.dataFirst(), data_size, cudaMemcpyHostToDevice, info->stream);
    cufftSetStream(plan, info->stream);
    cufftSetWorkArea(plan, info->work);
    auto status = cufftExecR2C(plan, reinterpret_cast<cufftReal *>(info->slab), reinterpret_cast<cufftComplex *>(info->slab));
    if (status != CUFFT_SUCCESS)
    {
        const char *errorString = cufftGetErrorString(status);
        printf("CUDA cuFFT Error: %s\n", errorString);
    }
    cudaMemcpyAsync(grid.dataFirst(), info->slab, data_size, cudaMemcpyDeviceToHost, info->stream);
}

