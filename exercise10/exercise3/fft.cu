#include <blitz/array.h>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <cuda.h>
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

int main()
{
    int n = 10000;

   Array<float, 2> raw_data3(n, n + 2);
    Array<float, 2> rdata3 = raw_data3(Range(0, n - 1), Range(0, n - 1));
    fill_array(rdata3);
    // Calculate the size in bytes
    size_t sizeInBytes = n * (n + 2) * sizeof(float);
    // Allocate memory on the GPU
    void *deviceData;
    cudaMalloc(&deviceData, sizeInBytes);
    // Copy data from CPU to GPU
    cudaMemcpy(deviceData, rdata3.data(), sizeInBytes, cudaMemcpyHostToDevice);

    Array<float, 2> raw_data4(n, n + 2);
    Array<float, 2> rdata4 = raw_data4(Range(0, n - 1), Range(0, n - 1));

    cudaMemcpy(rdata4.data(), deviceData, sizeInBytes, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    std::cout << (all(abs(rdata3 - rdata4)) < 1e-5 ? "match" : "mismatch") << std::endl;
    // Free GPU memory
    cudaFree(deviceData);

    return 0;
}
