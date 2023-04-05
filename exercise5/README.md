# Advanced High Performance Computing Exercise Sheet 5

## Exercise 1 [Simple Test]

```
    // Simple test
    float grid_sum = sum(grid(Range::all(), Range::all(), Range::all()));
    cout << "Sum of all grid mass = " << grid_sum << std::endl;
```

## Exercise 2 [Unit Test]
Unit tests are in `unit_test/` folder.

## Exercise 3 [System Integration]
Integration of unit tested kernels in `mass_assignment/`