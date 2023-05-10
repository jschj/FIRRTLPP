#pragma once

#include "firp.hpp"


namespace ufloat {

struct UFloatConfig {
  uint32_t exponentWidth;
  uint32_t mantissaWidth;
};

class PipelinedAdder : public firp::Module<PipelinedAdder> {
  uint32_t bitWidth, maxAdderWidth;
public:
  PipelinedAdder(uint32_t bitWidth, uint32_t maxAdderWidth):
    firp::Module<PipelinedAdder>(
      "PipelinedAdder",
      {
        firp::Port("a", true, firp::uintType(bitWidth)),
        firp::Port("b", true, firp::uintType(bitWidth)),
        firp::Port("c", false, firp::uintType(bitWidth + 1))
      },
      bitWidth, maxAdderWidth
    ), bitWidth(bitWidth), maxAdderWidth(maxAdderWidth) { build(); }
  
  void body();

  static uint32_t getDelay(uint32_t bitWidth, uint32_t maxAdderWidth) {
    return bitWidth / maxAdderWidth + (bitWidth % maxAdderWidth ? 1 : 0);
  }
};

}