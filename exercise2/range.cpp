#include <iostream>
#include "blitz/array.h"
using namespace blitz;
int main() {
    Array<int, 3> A(8,8,4);
    A = 7;
    std::cout << A << std::endl;
    Array<int,3> B = A(Range(5,7), Range(5,7), Range(0,2));
    B = 4;
    std::cout << A << std::endl;
}