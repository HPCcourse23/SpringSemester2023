#include <blitz/array.h>

int main() {
  // create two arrays with the same dimensions
  blitz::Array<double, 2> A(3, 3);
  blitz::Array<double, 2> B(3, 3);

  // initialize the arrays with some values
  A = 1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0;

  B = 9.0, 8.0, 7.0,
      6.0, 5.0, 4.0,
      3.0, 2.0, 1.0;

  // add the arrays together
  blitz::Array<double, 2> C(3, 3);
  C = A + B;

  // print the resulting array
  std::cout << C << std::endl;

  return 0;
}
