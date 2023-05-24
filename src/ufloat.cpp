#include "ufloat.hpp"




namespace ufloat {

using namespace firp;

FIRRTLBaseType UFloatConfig::getType() const {
  return ufloatType(*this);
}

FIRRTLBaseType ufloatType(const UFloatConfig& cfg) {
  return bundleType({
    {"e", false, uintType(cfg.exponentWidth)},
    {"m", false, uintType(cfg.mantissaWidth)}
  });
}

std::tuple<FValue, FValue> swapIfGTE(FValue a, FValue b) {
  auto doSwap = a("e") >= b("e");
  return std::make_tuple(
    regNext(mux(doSwap, a, b)),
    regNext(mux(doSwap, b, a))
  );
}

std::tuple<FValue, FValue> subtractStage(FValue minuend, FValue subtrahend) {
  auto difference = regNext(minuend - subtrahend);
  auto isZero = regNext(minuend == cons(0));
  return std::make_tuple(difference, isZero);
}

FValue shiftStage(FValue value, FValue shamt) {
  return regNext(cat({cons(1, bitType()), value}) >> shamt);
}

std::tuple<FValue, FValue> mantissaShift(FValue sum, uint32_t manWidth) {
  uint32_t hi = sum.bitCount() - 1;
  
  auto shifted = mux(
    sum(hi),
    sum(hi - 1, 0)(manWidth, 1),
    sum(hi - 2, 0)
  );

  assert(shifted.bitCount() == manWidth);

  return std::make_tuple(
    shifted,
    sum(hi)
  );
}

FValue exponentAdder(FValue eIn, FValue doAdd, FValue inputsAreZero) {
  return regNext(
    mux(doAdd & ~inputsAreZero, eIn + cons(1, bitType()), eIn)
  );
}

FValue uintToUFloat(FValue what, const UFloatConfig& cfg) {
  return bundleCreate(
    cfg.getType().dyn_cast<BundleType>(),
    {
      what(cfg.getWidth() - 1, cfg.mantissaWidth),
      what(cfg.mantissaWidth - 1, 0)
    }
  );
}

void FPAdd::body() {

#define SHOW(what) wireInit(what, #what)

  // stage 1: swap
  auto [a, b] = swapIfGTE(
    uintToUFloat(regNext(io("a")), cfg),
    uintToUFloat(regNext(io("b")), cfg)
  );

  wireInit(a, "a_uf");
  wireInit(b, "b_uf");

  // stage 2: subtract mantissas
  auto [difference_2, isZero_2] = subtractStage(a("e"), b("e"));
  auto m1_2 = regNext(a("m"));
  auto m2_2 = regNext(b("m"));
  auto e1_2 = regNext(a("e"));
  auto e2_2 = regNext(b("e"));

  SHOW(difference_2);
  SHOW(isZero_2);
  SHOW(m1_2);
  SHOW(m2_2);
  SHOW(e1_2);
  SHOW(e2_2);

  // stage 3: shift smaller mantissas
  auto m1_3 = regNext(m1_2);
  auto m2_3 = shiftStage(m2_2, difference_2);
  auto e1_3 = regNext(e1_2);
  auto minuend_is_zero_3 = regNext(isZero_2);

  SHOW(m1_3);
  SHOW(m2_3);
  SHOW(e1_3);
  SHOW(minuend_is_zero_3);

  // stage 4: add mantissas
  PipelinedAdder adder(cfg.mantissaWidth + 1, 32);
  adder.io("a") <<= cat({cons(1, bitType()), m1_3});
  adder.io("b") <<= m2_3;

  auto mantissaSum = adder.io("c");
  auto minuend_is_zero_4 = shiftRegister(minuend_is_zero_3, adder.getDelay());
  auto e1_4 = shiftRegister(e1_3, adder.getDelay());

  SHOW(mantissaSum);

  // stage 5: shift sum if necessary and increment exponent
  auto [shiftedMantissaOut, shiftedMantissaShifted] = mantissaShift(mantissaSum, cfg.mantissaWidth);
  auto m_6 = regNext(shiftedMantissaOut);

  SHOW(shiftedMantissaOut);
  SHOW(shiftedMantissaShifted);

  io("c") <<= cat({
    exponentAdder(e1_4, shiftedMantissaShifted, minuend_is_zero_4),
    m_6
  });
}

/*
FValue DSPMult24x17(FValue x, FValue y) {
  assert(x.bitCount() == 24);
  assert(y.bitCount() == 17);
  return regNext(x * y);
}

FValue SmallMult(FValue x, FValue y) {
  return regNext(x * y);
}

FValue Add(FValue x, FValue y, uint32_t width) {
  PipelinedAdder adder(width, 32);
  adder.io("a") <<= x;
  adder.io("b") <<= y;
  return adder.io("c");
}

FValue treeReduce(const std::vector<FValue>& values, uint32_t from, uint32_t to, uint32_t adderDelay) {
  // TODO: What about the bit widths of the values?

  if (to - from == 1)
    return shiftRegister(values[from], adderDelay);

  if (to - from == 2)
    return Add(values[from], values[from + 1], values[from].bitCount());

  uint32_t mid = (from + to) / 2;

  auto left = treeReduce(values, from, mid, adderDelay);
  auto right = treeReduce(values, mid, to, adderDelay);

  return Add(left, right, left.bitCount());
}

void DSPMult::body() {

  // TODO: How to get allocations?
  std::vector<DSPMultAllocation> allocations;

  std::vector<FValue> dsps;

  for (const DSPMultAllocation& alloc : allocations) {
    uint32_t xLo = alloc.xSelectorA();
    uint32_t xHi = std::min(width - 1, alloc.xSelectorB());
    uint32_t yLo = alloc.ySelectorA();
    uint32_t yHi = std::min(width - 1, alloc.ySelectorB());
    // TODO: Check if upper bound is exclusive.
    uint32_t xIn = io("a")(xHi, xLo);
    uint32_t yIn = io("b")(yHi, yLo);

    // TODO: check logical precedence
    if (alloc.xWidth + alloc.yWidth == 41 && alloc.xWidth == 24 || alloc.yWidth == 24) {
      auto res = alloc.xWidth == 24 ? DSPMult24x17(xIn, yIn) : DSPMult24x17(yIn, xIn);
      auto shifted = alloc.shift() != 0 ? cat({res, cons(0, uintType(alloc.shift()))}) : res;
      dsps.push_back(shifted);
    } else {
      auto res = SmallMult(xIn, yIn); // TODO: Maybe add alloc.xWidth and alloc.yWidth?
      auto shifted = alloc.shift() != 0 ? cat({res, cons(0, uintType(alloc.shift()))}) : res;
      dsps.push_back(shifted);
    }
  }

  io("c") <<= treeReduce(dsps, 0, dsps.size(), 123);
}*/

}