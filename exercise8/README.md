# Advanced High Performance Computing Exercise Sheet 8

## Exercise 1 [Partial Grid]
To create grid $local_0 * n_1 * n_2$ you need to use 
```
    float *data = new (std::align_val_t(64)) float[local0 * nGrid * (nGrid + 2)];
    blitz::GeneralArrayStorage<3> storage;
    storage.base() = start0, 0, 0;
    blitz::Array<float, 3> grid_data(data, blitz::shape(local0, nGrid, nGrid + 2), blitz::deleteDataWhenDone, storage);
    grid_data = 0.0;
    blitz::Array<float, 3> grid = grid_data(blitz::Range(start0, std::min(int(start0 + local0), nGrid) - 1), blitz::Range::all(), blitz::Range(0, nGrid - 1));
    std::complex<float> *complex_data = reinterpret_cast<std::complex<float> *>(data);
    blitz::Array<std::complex<float>, 3> kdata(complex_data, blitz::shape(local0, nGrid, nGrid / 2 + 1));
```
## For next exercises
Code in `mass_assignment/`
