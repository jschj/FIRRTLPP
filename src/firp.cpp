#include "firp.hpp"


namespace firp {

FirpContext::FirpContext(MLIRContext *ctxt, const std::string& topModule): ctxt(ctxt), opBuilder(ctxt) {
  root = builder().create<ModuleOp>(
    builder().getUnknownLoc()
  );

  builder().setInsertionPointToStart(
    &root.getBodyRegion().front()
  );

  circuitOp = builder().create<CircuitOp>(
    builder().getUnknownLoc(),
    builder().getStringAttr(topModule)
  );

  opBuilder = circuitOp.getBodyBuilder();
}

void FirpContext::beginContext(Value clock, OpBuilder bodyBuilder) {
  builderStack.push(opBuilder);
  opBuilder = bodyBuilder;

  clockStack.push(this->clock);
  this->clock = clock;
}

void FirpContext::endContext() {
  opBuilder = builderStack.top();
  builderStack.pop();

  clock = clockStack.top();
  clockStack.pop();  
}

static std::unique_ptr<FirpContext> ctxt;

FirpContext *firpContext() {
  return ctxt.get();
}

void initFirpContext(MLIRContext *mlirCtxt, const std::string& topModule) {
  ctxt = std::make_unique<FirpContext>(mlirCtxt, topModule);
}

FValue lift(Value val) {
  return FValue(val);
}

FValue cons(uint64_t n, IntType type) {
  ::llvm::APInt concreteValue =
    type.getWidthOrSentinel() == -1 ?
    ::llvm::APInt(64, n) :
    ::llvm::APInt(type.getWidthOrSentinel(), n);

  Value val = ctxt->builder().create<ConstantOp>(
    ctxt->builder().getUnknownLoc(),
    type,
    concreteValue
  ).getResult();

  return lift(val);
}

FValue mux(FValue cond, FValue pos, FValue neg) {
  return lift(
    firpContext()->builder().create<MuxPrimOp>(
      firpContext()->builder().getUnknownLoc(),
      cond,
      pos,
      neg
    )
  );
}

FValue mux(FValue sel, std::initializer_list<FValue> options) {

}

FValue FValue::operator~() {
  return firpContext()->builder().create<NotPrimOp>(firpContext()->builder().getUnknownLoc(), *this).getResult();
}

FValue FValue::operator+(FValue other) {
  return lift(
    firpContext()->builder().create<AddPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator-(FValue other) {
  return lift(
    firpContext()->builder().create<SubPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator&(FValue other) {
  return lift(
    firpContext()->builder().create<AndPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator|(FValue other) {
  return lift(
    firpContext()->builder().create<OrPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator>(FValue other) {
  return lift(
    firpContext()->builder().create<GTPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator>=(FValue other) {
  return lift(
    firpContext()->builder().create<GEQPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator<(FValue other) {
  return lift(
    firpContext()->builder().create<LTPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator<=(FValue other) {
  return lift(
    firpContext()->builder().create<LEQPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator==(FValue other) {
  return lift(
    firpContext()->builder().create<EQPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator!=(FValue other) {
  return lift(
    firpContext()->builder().create<NEQPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator()(size_t hi, size_t lo) {
  return firpContext()->builder().create<BitsPrimOp>(
    firpContext()->builder().getUnknownLoc(),
    *this,
    firpContext()->builder().getI32IntegerAttr(hi),
    firpContext()->builder().getI32IntegerAttr(lo)
  ).getResult();
}

FValue FValue::operator()(size_t index) {
  return (*this)(index, index);
}

FValue FValue::operator()(const std::string& fieldName) {
  return firpContext()->builder().create<SubfieldOp>(
    firpContext()->builder().getUnknownLoc(),
    *this,
    fieldName
  ).getResult();
}

void FValue::operator<<=(FValue other) {
  firpContext()->builder().create<StrictConnectOp>(
    firpContext()->builder().getUnknownLoc(),
    *this,
    other
  );
}

Reg::Reg(FIRRTLBaseType type): type(type) {
  regOp = firpContext()->builder().create<RegOp>(
    firpContext()->builder().getUnknownLoc(),
    type,
    firpContext()->getClock()
  );
}

void Reg::write(FValue what) {
  FValue input = what;

  // helps to simplify expressions such as reg.write(reg.read() + cons(1))
  if (llvm::dyn_cast<IntType>(type) && llvm::dyn_cast<IntType>(what.getType())) {
    int32_t hi = type.getBitWidthOrSentinel() - 1;
    assert(hi >= 0);
    input = what(hi, 0);
  }

  firpContext()->builder().create<StrictConnectOp>(
    firpContext()->builder().getUnknownLoc(),
    regOp,
    input
  );
}

void Conditional::build(size_t i, OpBuilder builder) {
  firpContext()->beginContext(firpContext()->getClock(), builder);

  bool isLast = i == cases.size() - 1;
  auto [cond, bodyCtor] = cases[i];

  WhenOp whenOp = firpContext()->builder().create<WhenOp>(
    firpContext()->builder().getUnknownLoc(),
    cond,
    !isLast || otherwiseCtor.has_value()
  );

  firpContext()->beginContext(firpContext()->getClock(), whenOp.getThenBodyBuilder());
  bodyCtor();
  firpContext()->endContext();

  if (!isLast) {
    build(i + 1, whenOp.getElseBodyBuilder());
  } else if (otherwiseCtor.has_value()) {
    firpContext()->beginContext(firpContext()->getClock(), whenOp.getElseBodyBuilder());
    otherwiseCtor.value()();
    firpContext()->endContext();
  }

  firpContext()->endContext();
}

Conditional::Conditional(FValue cond, BodyCtor bodyCtor) {
  cases.push_back(std::make_tuple(cond, bodyCtor));
}

Conditional& Conditional::elseWhen(FValue cond, BodyCtor bodyCtor) {
  cases.push_back(std::make_tuple(cond, bodyCtor));
  return *this;
}

Conditional& Conditional::otherwise(BodyCtor bodyCtor) {
  otherwiseCtor = bodyCtor;
  return *this;
}

Conditional::~Conditional() {
  build(0, firpContext()->builder());
}

Conditional when(FValue cond, Conditional::BodyCtor bodyCtor) {
  return Conditional(cond, bodyCtor);
}

IntType uintType() {
  return IntType::get(firpContext()->context(), false);
}

IntType uintType(uint32_t bitWidth) {
  return IntType::get(firpContext()->context(), false, bitWidth);
}

IntType bitType() {
  return uintType(1);
}

BundleType bundleType(std::initializer_list<std::tuple<std::string, bool, FIRRTLBaseType>> elements) {
  std::vector<BundleType::BundleElement> els;
  for (const auto& [name, flip, type] : elements)
    els.push_back(BundleType::BundleElement(
      firpContext()->builder().getStringAttr(name), flip, type
    ));

  return BundleType::get(els, firpContext()->context());
}

BundleType readyValidType(FIRRTLBaseType elementType) {
  return bundleType({
    {"ready", true, bitType()},
    {"valid", false, bitType()},
    {"bits", false, elementType}
  });
}

BundleType memReadType(FIRRTLBaseType dataType, uint32_t addrBits) {
  return bundleType({
    {"addr", false, uintType(addrBits)},
    {"en", false, bitType()},
    {"clk", false, ClockType::get(firpContext()->context())},
    {"data", true, dataType}
  });
}

BundleType memWriteType(FIRRTLBaseType dataType, uint32_t addrBits) {
  //int32_t bitWidth = dataType.getBitWidthOrSentinel();  
  //assert(bitWidth >= 0 && "cannot have uninferred widths in memory port type");
  //assert(bitWidth % 8 == 0 && "bit width must be divisible by 8");
  int32_t maskBits = 1; //bitWidth / 8;

  return bundleType({
    {"addr", false, uintType(addrBits)},
    {"en", false, bitType()},
    {"clk", false, ClockType::get(firpContext()->context())},
    {"data", false, dataType},
    {"mask", false, uintType(maskBits)}
  });
}

Memory::Memory(FIRRTLBaseType dataType, size_t depth) {
  size_t addrBits = clog2(depth);

  TypeRange resultTypes{
    memWriteType(dataType, addrBits),
    memReadType(dataType, addrBits)
  };

  std::vector<Attribute> portNames{
    firpContext()->builder().getStringAttr("writePort"),
    firpContext()->builder().getStringAttr("readPort")
  };

  memOp = firpContext()->builder().create<MemOp>(
    firpContext()->builder().getUnknownLoc(),
    resultTypes,
    0, 1, // r/w latency
    depth,
    RUWAttr::Undefined,
    ArrayRef(portNames)
  );
}

FValue Memory::writePort() {
  return memOp.getResults()[0];
}

FValue Memory::readPort() {
  return memOp.getResults()[1];
}

}