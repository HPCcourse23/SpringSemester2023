// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <cstdint>
#include <stdlib.h>
#include <chrono>
#include "blitz/array.h"
#include "tipsy.h"
#include <math.h>
#include <new>
#include <fftw3.h>

using namespace blitz;

int wrap_edge(int coordinate, int N)
{
    if (coordinate < 0)
    {
        coordinate += N;
    }
    else if (coordinate >= N)
    {
        coordinate -= N;
    }
    return coordinate;
}

float pow_3(float base)
{
    return base * base * base;
}

int precalculate_W(float W[], int order, float r, float cell_half = 0.5)
{
    int start = floorf(r - cell_half * (order - 1));

    float s[order];
    for (int i = start; i < start + order; i++)
        s[i - start] = abs(i + cell_half - r);
    switch (order)
    {
    case 1:
        W[0] = 1;
        return start;
    case 2:
        W[0] = 1.0 - s[0];
        W[1] = 1.0 - s[1];
        return start;
    case 3:
        W[0] = 0.5 * (1.5 - s[0]) * (1.5 - s[0]);
        W[1] = 0.75 - s[1] * s[1];
        W[2] = 0.5 * (1.5 - s[2]) * (1.5 - s[2]);
        return start;
    case 4:
        W[0] = 1.0 / 6.0 * pow_3(2.0 - s[0]);
        W[1] = 1.0 / 6.0 * (4.0 - 6.0 * s[1] * s[1] + 3 * pow_3(s[1]));
        W[2] = 1.0 / 6.0 * (4.0 - 6.0 * s[2] * s[2] + 3 * pow_3(s[2]));
        W[3] = 1.0 / 6.0 * pow_3(2.0 - s[3]);
        return start;
    default:
        std::cerr << "[precalculate_W] Out of bound " << endl;
        return -1;
    }
}

void assign_mass(Array<float, 2> &r, int N, int nGrid, Array<float, 3> &grid, int order = 4)
{
    // Loop over all cells for this assignment
    float cell_half = 0.5;
    std::cout << "Assigning mass to the grid using order " << order << std::endl;
#pragma omp parallel for
    for (int pn = 0; pn < N; ++pn)
    {
        float x = r(pn, 0);
        float y = r(pn, 1);
        float z = r(pn, 2);

        float rx = (x + 0.5) * nGrid;
        float ry = (y + 0.5) * nGrid;
        float rz = (z + 0.5) * nGrid;

        // precalculate Wx, Wy, Wz and return start index
        float Wx[order], Wy[order], Wz[order];
        int i_start = precalculate_W(Wx, order, rx);
        int j_start = precalculate_W(Wy, order, ry);
        int k_start = precalculate_W(Wz, order, rz);

        for (int i = i_start; i < i_start + order; i++)
        {
            for (int j = j_start; j < j_start + order; j++)
            {
                for (int k = k_start; k < k_start + order; k++)
                {
                    float W_res = Wx[i - i_start] * Wy[j - j_start] * Wz[k - k_start];

// Deposit the mass onto grid(i,j,k)
#pragma omp atomic
                    grid(wrap_edge(i, nGrid), wrap_edge(j, nGrid), wrap_edge(k, nGrid)) += W_res;
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc > 2)
        nGrid = atoi(argv[2]);

    int order = 4;
    if (argc > 3)
        order = atoi(argv[3]);

    const char *out_filename = (argc > 4) ? argv[4] : "projected.csv";

    auto start_time = std::chrono::high_resolution_clock::now();

    TipsyIO io;
    io.open(argv[1]);
    if (io.fail())
    {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }
    std::uint64_t N = io.count();

    // Load particle positions
    std::cerr << "Loading " << N << " particles" << std::endl;
    Array<float, 2> r(N, 3);
    io.load(r);

    std::chrono::duration<double> diff_load = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Reading file took " << std::setw(9) << diff_load.count() << " s\n";

    start_time = std::chrono::high_resolution_clock::now();

    float *data = new (std::align_val_t(64)) float[nGrid * nGrid * (nGrid + 2)];
    // Create Mass Assignment Grid
    Array<float, 3> grid_data(data, shape(nGrid, nGrid, nGrid), deleteDataWhenDone);
    Array<float, 3> grid = grid_data(Range::all(), Range::all(), Range(0, nGrid - 1));
    std::complex<float> *complex_data = reinterpret_cast<std::complex<float> *>(data);
    blitz::Array<std::complex<float>, 3> kdata(complex_data, shape(nGrid, nGrid, nGrid / 2 + 1));
    assign_mass(r, N, nGrid, grid, order);

    std::chrono::duration<double> diff_assignment = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Mass assignment took " << std::setw(9) << diff_assignment.count() << " s\n";

    start_time = std::chrono::high_resolution_clock::now();
    Array<float, 2> projected(nGrid, nGrid);
    for (int i = 0; i < nGrid; ++i)
    {
        for (int j = 0; j < nGrid; ++j)
        {
            projected(i, j) = max(grid(i, j, Range::all()));
        }
    }

    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Projection took " << std::setw(9) << diff.count() << " s\n";

    ofstream f(out_filename);
    for (int i = 0; i < nGrid; ++i)
    {
        for (int j = 0; j < nGrid; ++j)
        {
            if (j != nGrid - 1)
                f << projected(i, j) << ",";
            else
                f << projected(i, j);
        }
        f << endl;
    }
    f.close();


    fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, FFTW_ESTIMATE);
    cout << "Plan created" << endl;
    fftwf_execute(plan);
    cout << "Plan executed" << endl;
    fftwf_destroy_plan(plan);
    cout << "Plan destroyed" << endl;
}