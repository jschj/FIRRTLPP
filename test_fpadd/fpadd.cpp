#include <iostream>


int main() {
  double a = 76.6786117553711;
  double b = 44.01980972290039;
  double c = a + b;
  
  float fa = float(a);
  float fb = float(b);
  float fc = fa + fb;

  std::cout.precision(20);
  std::cout << "double: " << c << "\n";
  std::cout << "float: " << fc << "\n";

  return 0;
}