#include "blitz/array.h"
#include <iostream>

int main () {
    blitz::Array <float, 3> data (10, 8, 6);
    std::cout << data;
}
