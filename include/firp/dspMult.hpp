#pragma once

#include "pipelinedAdder.hpp"


namespace ufloat {

struct DSPTile {
  // These bounds are inclusive!
  uint32_t xHi, xLo, yHi, yLo;

  uint32_t getShift() const {
    return xLo + yLo;
  }
};

std::vector<DSPTile> getDSPTiles(uint32_t bitWidth);

class DSPMult : public firp::Module<DSPMult> {
  uint32_t width;
  uint32_t adderWidth;
  uint32_t maxAdderSize;
  uint32_t adderDelay;

  uint32_t moduleDelay;
public:
  DSPMult(uint32_t width):
    firp::Module<DSPMult>(
      "DSPMult",
      {
        firp::Port("a", true, firp::uintType(width)),
        firp::Port("b", true, firp::uintType(width)),
        firp::Port("c", false, firp::uintType(width * 2))
      },
      width
    ), width(width), adderWidth(width * 2),
       maxAdderSize(32), adderDelay(PipelinedAdder::getDelay(adderWidth, maxAdderSize)) {
      build();
    }

  void body();

  uint32_t getDelay() const {
    return moduleDelay;
  }
private:
  firp::FValue Add(firp::FValue x, firp::FValue y);
  std::tuple<firp::FValue, uint32_t> treeReduce(const std::vector<firp::FValue>& values, uint32_t from, uint32_t to);
};

}