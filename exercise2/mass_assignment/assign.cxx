// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <cstdint>
#include <stdlib.h>
#include "blitz/array.h"
#include "tipsy.h"

using namespace blitz;

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

    // Calculate generally
    // For future it is ok to use precalculated values and assume that all x,y,z ∈ (0.5, 0,5)
    float max_x = max(r(Range(0, N), 0)), max_y = max(r(Range(0, N), 1)), max_z = max(r(Range(0, N), 2));
    float min_x = min(r(Range(0, N), 0)), min_y = min(r(Range(0, N), 1)), min_z = min(r(Range(0, N), 2));
    float step_x = (max_x - min_x) / nGrid, step_y = (max_y - min_y) / nGrid, step_z = (max_z - min_z) / nGrid;

    // Create Mass Assignment Grid
    Array<float, 3> grid(nGrid, nGrid, nGrid);

    grid = 0;
    for (int pn = 0; pn < N; ++pn)
    {
        float x = r(pn, 0);
        float y = r(pn, 1);
        float z = r(pn, 2);

        // Convert x, y and z into a grid position i,j,k such that
        // 0 <= i < nGrid
        // 0 <= j < nGrid
        // 0 <= k < nGrid
        int i = (x - min_x) / step_x;
        int j = (y - min_y) / step_y;
        int k = (z - min_z) / step_z;

        // Deposit the mass onto grid(i,j,k)
        grid(i, j, k)++;
    }

    Array<float, 2> projected(nGrid, nGrid);
    for (int i = 0; i < nGrid; ++i)
    {
        for (int j = 0; j < nGrid; ++j)
        {
            projected(i, j) = max(grid(i, j, Range::all()));
        }
    }

    ofstream f("projected.csv");
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
}
