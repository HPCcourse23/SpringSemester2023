#include <iostream>
#include "blitz/array.h"
using namespace blitz;
int main() {
    GeneralArrayStorage <3> storage ;
    storage.base() = 10, 0, 0;

    Array<int, 3> A(5, 20, 20, storage);
    A = 100;
    A.dumpStructureInformation(cerr);
    std::cout << A(11, 0, 0) << std::endl;
}