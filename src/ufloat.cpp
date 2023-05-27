#include <firp/ufloat.hpp>




namespace ufloat {

using namespace firp;

BundleType UFloatConfig::getType() const {
  return ufloatType(*this);
}

BundleType ufloatType(const UFloatConfig& cfg) {
  return bundleType({
    {"e", false, uintType(cfg.exponentWidth)},
    {"m", false, uintType(cfg.mantissaWidth)}
  });
}

FValue ufloatUnpack(FValue what, const UFloatConfig& cfg) {
  assert(what.bitCount() == cfg.getWidth());
  Value e = what(cfg.getWidth() - 1, cfg.mantissaWidth);
  Value m = what(cfg.mantissaWidth - 1, 0);
  return bundleCreate(cfg.getType(), {e, m});
}

FValue ufloatPack(FValue what, const UFloatConfig& cfg) {
  assert(what.bitCount() == cfg.getWidth());
  return cat({what("e"), what("m")});
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

FValue isEitherZero(FValue a, FValue b) {
  uint32_t m = a("m").bitCount();
  uint32_t e = a("e").bitCount();

  return (a("e") == uval(0, e) & a("m") == uval(0, m)) |
         (b("e") == uval(0, e) & b("m") == uval(0, m));
}

FValue addExponents(FValue e1, FValue e2) {
  return regNext(e1 + e2);
}

std::tuple<FValue, FValue> incrementExponentAndSubtractOffset(FValue e, FValue add) {
  uint32_t off = (1 << (e.bitCount() - 2)) - 1;
  auto offsetDontAdd = sval(off);
  auto offsetAdd = sval(off - 1);
  auto e1Signed = cat({uval(0), e}).asSInt();
  uint32_t n = e1Signed.bitCount();

  auto selected = mux(add, e1Signed - offsetAdd, e1Signed - offsetDontAdd).tail(1); // throw away the +1 bit
  auto overflowOut = selected.head(1);
  // TODO: Why do the bit widths not match without head(8)?
  auto eOut = selected.tail(2); //.head(8);

  return std::make_tuple(eOut, overflowOut);
}

FValue selectMantissa(FValue m, uint32_t manWidth) {
  auto geq2 = m.head(1);

  return mux(
    geq2,
    m.tail(1).head(manWidth),
    m.tail(2).head(manWidth)
  );
}

FValue handleZero(FValue m, FValue e, FValue underflow, FValue zero) {
  auto mOut = regNext(mux(zero | underflow, uval(0, m.bitCount()), m));
  auto eOut = regNext(mux(zero | underflow, uval(0, e.bitCount()), e));
  return cat({eOut, mOut});
}

void FPMult::body() {
  auto op1 = regNext(ufloatUnpack(io("a"), cfg));
  auto op2 = regNext(ufloatUnpack(io("b"), cfg));

  // stage 1
  DSPMult multMantissas(cfg.mantissaWidth + 1);
  multMantissas.io("a") <<= cat({uval(1), op1("m")});
  multMantissas.io("b") <<= cat({uval(1), op2("m")});

  auto zeroCheck = isEitherZero(op1, op2);
  auto addExponentsWithOffset = addExponents(op1("e"), op2("e"));

  // stage 2
  uint32_t dspDelay = multMantissas.getDelay();
  auto [eOut, underflowOut] = incrementExponentAndSubtractOffset(
    shiftRegister(addExponentsWithOffset, dspDelay - 1),
    multMantissas.io("c").head(1)
  );

  auto result = handleZero(
    selectMantissa(multMantissas.io("c"), cfg.mantissaWidth),
    eOut,
    underflowOut,
    shiftRegister(zeroCheck, dspDelay - 1)
  );

  io("c") <<= result;
}

void FPConvert::body() {
  // TODO: implement
  io("out") <<= io("in");
}

}

#include <firp/lowering.hpp>

using namespace ::firp;
using namespace ::circt::firrtl;
using namespace ::mlir;

void generateFPAdd() {
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();
  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());
  assert(context->getOrLoadDialect<circt::sv::SVDialect>());

  initFirpContext(context.get(), "FPAdd");

  {
    ufloat::FPAdd add(ufloat::UFloatConfig{8, 23});
    add.makeTop();
  }

  firpContext()->finish();
  firpContext()->dump();

  assert(succeeded(lowerFirrtlToHw()));
  assert(succeeded(exportVerilog(".")));
}

void generateDSPMult() {

}

void generateFPMult() {
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();
  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());
  assert(context->getOrLoadDialect<circt::sv::SVDialect>());

  initFirpContext(context.get(), "FPMult");

  {
    ufloat::FPMult mult(ufloat::UFloatConfig{8, 23});
    mult.makeTop();
  }

  firpContext()->finish();
  firpContext()->dump();

  assert(succeeded(lowerFirrtlToHw()));
  assert(succeeded(exportVerilog(".")));
}

int main(int argc, const char **argv) {
  generateFPAdd();
  generateFPMult();

  return 0;
}