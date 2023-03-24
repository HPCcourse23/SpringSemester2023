# Advanced High Performance Computing Exercise Sheet 4

All code in `mass_assignment/`

## Exercise 1 [Memory Allocation]

```
float *data = new(std::align_val_t(64)) float[nGrid * nGrid * nGrid];
Array <float, 3> grid_data(
    data, 
    shape(nGrid, nGrid, nGrid), 
    deleteDataWhenDone
);
```

## Exercise 2 [In-place Memory Order]
```
float *data = new(std::align_val_t(64)) float[nGrid * nGrid * (nGrid + 2)];
Array <float, 3> grid_data(
    data, 
    shape(nGrid, nGrid, nGrid + 2), 
    deleteDataWhenDone
);
```

## Exercise 3 [Subarrays]

```
Array <float, 3> grid = grid_data(
    Range::all(), 
    Range::all(), 
    Range(0 , nGrid - 1)
);
```

## Exercise 4 [Complex Array]

```
std::complex <float>* complex_data = reinterpret_cast<std::complex <float>*>(data);
blitz::Array <std::complex<float>, 3> kdata (
    complex_data,
    shape(nGrid, nGrid, nGrid / 2 + 1)
);
```

## Exercise 5 [FFT]

```
fftwf_plan plan = fftwf_plan_dft_r2c_3d(
    nGrid,
    nGrid,
    nGrid,
    data,
    reinterpret_cast <fftwf_complex*>(complex_data),
    FFTW_ESTIMATE
);
fftwf_execute(plan);
fftwf_destroy_plan(plan);
```
