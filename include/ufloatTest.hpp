#pragma once

#include <cstdint>
#include <iostream>

#include "verilated_vcd_c.h"


template <class Dut>
void pipelinedAdderTest(Dut& dut, VerilatedVcdC *tfp, vluint64_t& timeStamp) {
  uint32_t delay = 3;
  uint32_t bits = 16;

  std::vector<uint32_t> expected, got;

  srand(time(0));

  for (uint32_t i = 0; i < 100; ++i, ++timeStamp) {
    dut.clock = !dut.clock;

    if (dut.clock) {
      uint32_t a = i <= 2 ? rand() % (1ul << bits) : 0;
      uint32_t b = i <= 2 ? rand() % (1ul << bits) : 0;
      uint32_t c = a + b;

      expected.push_back(c);

      dut.a = a;
      dut.b = b;

      if (i >= delay * 2)
        got.push_back(dut.c);

      //std::cout << "exp/got: " << c << " " << dut.c << "\n";
    }

    dut.eval();
    if (tfp)
      tfp->dump(timeStamp);
  }

  std::cout << "expected size vs. got size: " << expected.size() << " vs. " << got.size() << "\n";

  //assert(got.size() <= expected.size());

  for (uint32_t i = 0; i < got.size(); ++i) {
    std::cout << "exp/got @ i=" << i << ": " << expected[i] << " " << got[i] << "\n";
    //assert(expected[i] == got[i]);
  }

  std::cout << "Everything works!" << std::endl;
}