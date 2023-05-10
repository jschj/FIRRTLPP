#pragma once

#include "firp.hpp"


namespace firp::test {

using namespace ::mlir;
using namespace ::circt::firrtl;




template <class ConcreteModule>
class Test {
protected:
  Test() {
    // 1.
    // instantiate ConcreteModule with the parameters
    // make it top
    // export the verilog files
    // create c++ files from verilog files

    // 2.
    // 
  }

  template <class...Args>
  void initialize(Args...args) {
    
  }
  

};

template <class TestClass>
class RunTest {

};

void runTest(const std::string& className) {
  // 1.
  // assume firpContext()->finish() was already called
  // convert to verilog
  // export verilog files
  // create c++ files from verilog files (verilator --cc *.sv)

  // 2.
  // generate Makefile that compiles className.hpp
  // execute Makefile

  // 3.
  // verilate
}

}