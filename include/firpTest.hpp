#pragma once

#include "firp.hpp"


namespace firp::test {

using namespace ::mlir;
using namespace ::circt::firrtl;

class integer {
public:
  integer(uint64_t value) {

  }

  void operator=(uint64_t value) {

  }
};

template <class ConcreteTestBench>
class FirpTestBench {
  void declare() {

  }
protected:
  // create a always block that listens for positive clock edges
  template <class Ctor>
  void thread(Ctor bodyCtor) {
    
  }
public:

  FirpTestBench() {

  }

};

class MyTestBench : public FirpTestBench<MyTestBench> {


public:
  void test() {
    thread([&](){

    });
  }
};

}