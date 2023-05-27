#include "dspMult.hpp"


namespace ufloat {

using namespace firp;

// TODO: Is it 25x18 or 24x17?

struct DSPTile {
  // These bounds are inclusive!
  uint32_t xHi, xLo, yHi, yLo;

  uint32_t getShift() const {
    return xLo + yLo;
  }
};

// This is a really stupid way to tile a bitWidth x bitWidth grid with 25x18 DSPs only!
std::vector<DSPTile> getDSPTiles(uint32_t bitWidth) {
  std::vector<DSPTile> tiles;

  for (uint32_t xLo = 0; xLo < bitWidth; xLo += 25) {
    uint32_t xHi = std::min(xLo + 24, bitWidth - 1);
    for (uint32_t yLo = 0; yLo < bitWidth; yLo += 18) {
      uint32_t yHi = std::min(yLo + 17, bitWidth - 1);
      tiles.push_back({xHi, xLo, yHi, yLo});
    }
  }

  return tiles;
}

FValue DSPMult24x17(FValue x, FValue y) {
  assert(x.bitCount() <= 25);
  assert(y.bitCount() <= 18);
  auto result = regNext(x * y);
  wireInit(result, "DSPMult24x17");
  return result;
}

FValue DSPMult::Add(FValue x, FValue y) {
  PipelinedAdder adder(adderWidth, maxAdderSize);
  adder.io("a") <<= x;
  adder.io("b") <<= y;
  return adder.io("c");
}

std::tuple<firp::FValue, uint32_t> DSPMult::treeReduce(const std::vector<FValue>& values, uint32_t from, uint32_t to) {
  // TODO: What about the bit widths of the values?

  if (to - from == 1)
    return std::make_tuple(shiftRegister(values[from], adderDelay), adderDelay);

  if (to - from == 2)
    return std::make_tuple(Add(values[from], values[from + 1]), adderDelay);

  uint32_t mid = (from + to) / 2;

  auto [left, leftDelay] = treeReduce(values, from, mid);
  auto [right, rightDelay] = treeReduce(values, mid, to);

  return std::make_tuple(Add(left, right), std::max(leftDelay, rightDelay) + adderDelay);
}

void DSPMult::body() {
  auto tiles = getDSPTiles(width);

  std::vector<FValue> parts;

  for (const auto& tile : tiles) {
    auto x = io("a")(tile.xHi, tile.xLo);
    auto y = io("b")(tile.yHi, tile.yLo);
    auto zeros = cons(0, uintType(tile.getShift()));
    parts.push_back(cat({DSPMult24x17(x, y), zeros}));
  }

  auto [result, delay] = treeReduce(parts, 0, parts.size());
  moduleDelay = delay;

  io("c") <<= result;
}

}