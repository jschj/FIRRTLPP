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

void PipelinedAdder::body() {
  // inspired by http://vlsigyan.com/pipeline-adder-verilog-code/

  if (bitWidth <= maxAdderWidth) {
    io("c") <<= io("a") + io("b");
    return;
  }

  uint32_t stageCount = bitWidth / maxAdderWidth + (bitWidth % maxAdderWidth ? 1 : 0);
  auto carry = cons(0, uintType(1));
  //Wire result(uintType(bitWidth + 1));
  std::vector<FValue> resultChunks;

  for (uint32_t i = 0; i < stageCount; ++i) {
    uint32_t lo = i * maxAdderWidth;
    uint32_t hi = std::min(lo + maxAdderWidth - 1, bitWidth - 1);

    uint32_t preDelay = i;
    uint32_t postDelay = stageCount - i - 1;

    auto inA = wireInit(shiftRegister(io("a")(hi, lo), preDelay), std::string("inA_") + std::to_string(i));
    auto inB = wireInit(shiftRegister(io("b")(hi, lo), preDelay), std::string("inB_") + std::to_string(i));

    // top bit is ignored as it will never be set to 1
    auto sum = inA.read() + inB + shiftRegister(carry, 1);
    uint32_t sumWidth = sum.bitCount();
    llvm::outs() << " sum/inA/inB width: " << sum.bitCount() << " " << inA.read().bitCount() << " " << inB.read().bitCount() << "\n";

    // top bit will always be 0
    auto sumOut = sum(sumWidth - 3, 0);
    carry = sum(sumWidth - 2);

    resultChunks.push_back(shiftRegister(sumOut, postDelay));
  }

  resultChunks.push_back(carry);
  std::reverse(resultChunks.begin(), resultChunks.end());
  
  io("c") <<= cat(resultChunks);

  //svCocoTBVerbatim("PipelinedAdder");
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

std::tuple<FValue, FValue> mantissaShift(FValue sum) {
  uint32_t hi = sum.bitCount() - 1;
  return std::make_tuple(
    mux(sum(hi), sum(hi - 1, 0), sum(hi - 2, 0)),
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

  // stage 5: shift sum if necessary and increment exponent
  auto [shiftedMantissaOut, shiftedMantissaShifted] = mantissaShift(mantissaSum);
  auto m_6 = regNext(shiftedMantissaOut);

  io("c") <<= cat({
    exponentAdder(e1_4, shiftedMantissaShifted, minuend_is_zero_4),
    m_6
  });
}

}