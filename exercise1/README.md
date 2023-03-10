# Advanced High Performance Computing Exercise Sheet 1

## Exercise 1 [Arrays in C++]
(i) Which parts of line 3 `(blitz::Array<float,3> data(10,8,6))` correspond to **namespace**, **template**, and **constructor**?

    namespace: blitz
    template: Array<float, 3>
    constructor: data(10, 8, 6)

(ii) What do you think that this line does?

    Creates 3-dimensional array 10x8x6

Modify the first program to print “data” instead of 42.
```
#include "blitz/array.h"
#include <iostream>

int main () {
    blitz::Array <float, 3> data (10, 8, 6);
    std::cout << data;
}
```
## Exercise 2 [Memory Usage]
Let be N - the number of nodes, then

4036^3 = N * 64Gb/56b =>

1 Gb = 2^30b  =>

2^36 = N * 2^30 * 2^6 / 56 =>

N = 2^36 * 56 / 2^36 =>

N = 56

Answer: 56 nodes

## Exercise 3 [Tipsy Header]
(i) What is the simulation time of the file?

    1.0000000000000107

(ii) How many particles are in the file?

    158095

(iii) What od commands did you use to determine this?
```
(i)  od --endian=big -j 0 -N 8 -t f8 b0-final.std
(ii) od --endian=big -j 8 -N 4 -t d4 b0-final.std
```

## Exercise 4 [Tipsy Particles]
(i) What is the mass and position of particle 100?

    mass = 2.0393363e-09
    x = -0.008689348
    y = -0.03393134
    z = -0.03598262

(ii) What od commands did you use to determine this?

```
for mass =     od --endian=big -j 3596 -N 4 -t f4 b0-final.std
for position = od --endian=big -j 3600 -N 12 -t f4 b0-final.std
```
## Exercise 5 [Tipsy Analysis]

(i) What is the minimum and maximum x, y and z coordinates?

    X from -0.499911904335022 to 0.4999992549419403
    Y from -0.4998406171798706 to 0.49973100423812866
    Z from -0.49995994567871094 to 0.49990418553352356

(ii) What is the total mass in the box (sum of the mass of all particles).

    Total mass is 0.23699997134654538

Code in `readtipsy.py`
