#include <blitz/array.h>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cufft.h>
using namespace blitz;
using std::complex;

void fill_array(Array<float, 2> &data)
{
    // Set the grid to the sum of two sine functions
    for (int i = 0; i < data.rows(); i++)
    {
        for (int j = 0; j < data.cols(); j++)
        {
            float x = (float)i / 25.0; // Period of 1/4 of the box in x
            float y = (float)j / 10.0; // Period of 1/10 of the box in y
            data(i, j) = sin(2.0 * M_PI * x) + sin(2.0 * M_PI * y);
        }
    }
}

// Verify the FFT (kdata) of data by performing a reverse transform and comparing
bool validate(Array<float, 2> &data, Array<std::complex<float>, 2> kdata)
{
    Array<float, 2> rdata(data.extent());
    fftwf_plan plan = fftwf_plan_dft_c2r_2d(data.rows(), data.cols(),
                                            reinterpret_cast<fftwf_complex *>(kdata.data()), rdata.data(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    rdata /= data.size(); // Normalize for the FFT
    return all(abs(data - rdata) < 1e-5);
}
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

int main()
{
    int N = 10000;

    // Out of place
    Array<float, 2> rdata1(N, N);
    Array<std::complex<float>, 2> kdata1(N, N / 2 + 1);
    fftwf_plan plan1 = fftwf_plan_dft_r2c_2d(N, N,
                                             rdata1.data(), reinterpret_cast<fftwf_complex *>(kdata1.data()), FFTW_ESTIMATE);
    fill_array(rdata1);
    fftwf_execute(plan1);
    fftwf_destroy_plan(plan1);
    std::cout << ">>> Out of place FFT " << (validate(rdata1, kdata1) ? "match" : "MISMATCH") << endl;

    // in-place
    Array<float, 2> raw_data2(N, N + 2);
    Array<float, 2> rdata2 = raw_data2(Range(0, N - 1), Range(0, N - 1));
    fftwf_plan plan2 = fftwf_plan_dft_r2c_2d(N, N,
                                             rdata2.data(), reinterpret_cast<fftwf_complex *>(rdata2.data()), FFTW_ESTIMATE);
    fill_array(rdata2);
    fftwf_execute(plan2);
    fftwf_destroy_plan(plan2);
    Array<std::complex<float>, 2> kdata2(reinterpret_cast<std::complex<float> *>(rdata2.data()),
                                         shape(N, N / 2 + 1), neverDeleteData);
    std::cout << ">>> In-place FFT " << (validate(rdata1, kdata2) ? "match" : "MISMATCH") << endl;

    Array<float, 2> raw_data5(N, N + 2);
    Array<float, 2> rdata5 = raw_data2(Range(0, N - 1), Range(0, N - 1));
    auto data_size = sizeof(cufftComplex) * rdata5.rows() * (rdata5.cols() / 2 + 1);
    void *device_data;
    cudaMalloc(&device_data, data_size);

    fill_array(rdata5);

    int n[] = {N, N}; // 2D FFT of length NxN
    int inembed[] = {rdata5.rows(), 2 * (rdata5.cols() / 2 + 1)};
    int onembed[] = {rdata5.rows(), (rdata5.cols() / 2 + 1)};
    int batch = 1;
    int odist = rdata5.rows() * (rdata5.cols() / 2 + 1); // Output distance is in "complex"
    int idist = 2 * odist;                               // Input distance is in "real"
    int istride = 1;                                     // Elements of each FFT are adjacent
    int ostride = 1;
    cufftHandle plan;
    cufftPlanMany(&plan, sizeof(n) / sizeof(n[0]), n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_R2C, batch);

    cudaMemcpy(device_data, rdata5.dataFirst(), data_size, cudaMemcpyHostToDevice);

    auto status = cufftExecR2C(plan, reinterpret_cast<cufftReal *>(device_data), reinterpret_cast<cufftComplex *>(device_data));
    if (status != CUFFT_SUCCESS)
    {
        const char *errorString = cufftGetErrorString(status);
        printf("CUDA cuFFT Error: %s\n", errorString);
    }

    cudaMemcpy(rdata5.dataFirst(), device_data, data_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    Array<std::complex<float>, 2> kdata5(reinterpret_cast<std::complex<float> *>(rdata5.data()),
                                         shape(N, N / 2 + 1), neverDeleteData);
    std::cout << ">>> In-place GPU FFT " << (validate(rdata1, kdata5) ? "match" : "MISMATCH") << endl;

    cudaFree(device_data);
    cufftDestroy(plan);
    return 0;
}
