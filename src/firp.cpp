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

void FirpContext::beginContext(Value clock, Value reset, OpBuilder bodyBuilder) {
  builderStack.push(opBuilder);
  opBuilder = bodyBuilder;

  clockStack.push(this->clock);
  this->clock = clock;

  resetStack.push(this->reset);
  this->reset = reset;
}

void FirpContext::endContext() {
  opBuilder = builderStack.top();
  builderStack.pop();

  clock = clockStack.top();
  clockStack.pop();  

  reset = resetStack.top();
  resetStack.pop();  
}

void FirpContext::beginModuleDeclaration() {
  beginContext(getClock(), getReset(), circuitOp.getBodyBuilder());
}

void FirpContext::endModuleDeclaration() {
  endContext();
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
  std::vector<Value> asValues(std::cbegin(options), std::cend(options));

  return firpContext()->builder().create<MultibitMuxOp>(
    firpContext()->builder().getUnknownLoc(),
    sel,
    ValueRange(asValues)
  ).getResult();
}

FValue zeros(FIRRTLBaseType type) {
  int32_t width = type.getBitWidthOrSentinel();
  assert(width >= 0);
  FValue zeroValue = cons(0, uintType(width));

  if (llvm::isa<IntType>(type))
    return zeroValue;

  return firpContext()->builder().create<BitCastOp>(
    firpContext()->builder().getUnknownLoc(),
    type,
    zeroValue
  ).getResult();
}

FValue ones(FIRRTLBaseType type) {
  int32_t width = type.getBitWidthOrSentinel();
  assert(llvm::dyn_cast<BundleType>(type).getBitWidthOrSentinel() >= 0);
  assert(width >= 0);

  ::llvm::APInt concreteValue(width, 0, true);
  concreteValue -= 1;

  FValue onesValue = ctxt->builder().create<ConstantOp>(
    ctxt->builder().getUnknownLoc(),
    uintType(width),
    concreteValue
  ).getResult();

  if (llvm::isa<IntType>(type))
    return onesValue;

  return firpContext()->builder().create<BitCastOp>(
    firpContext()->builder().getUnknownLoc(),
    type,
    onesValue
  ).getResult();
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

FValue FValue::operator*(FValue other) {
  return lift(
    firpContext()->builder().create<MulPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
  );
}

FValue FValue::operator/(FValue other) {
  return lift(
    firpContext()->builder().create<DivPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult()
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

Reg::Reg(FIRRTLBaseType type, FValue resetValue, const std::string& name): type(type) {
  regOp = firpContext()->builder().create<RegResetOp>(
    firpContext()->builder().getUnknownLoc(),
    type,
    firpContext()->getClock(),
    firpContext()->getReset(),
    resetValue,
    name
  );
}

Reg::Reg(FIRRTLBaseType type, const std::string& name):
  Reg(type, zeros(type), name) {}

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
  firpContext()->beginContext(firpContext()->getClock(), firpContext()->getReset(), builder);

  bool isLast = i == cases.size() - 1;
  auto [cond, bodyCtor] = cases[i];

  WhenOp whenOp = firpContext()->builder().create<WhenOp>(
    firpContext()->builder().getUnknownLoc(),
    cond,
    !isLast || otherwiseCtor.has_value()
  );

  firpContext()->beginContext(firpContext()->getClock(), firpContext()->getReset(), whenOp.getThenBodyBuilder());
  bodyCtor();
  firpContext()->endContext();

  if (!isLast) {
    build(i + 1, whenOp.getElseBodyBuilder());
  } else if (otherwiseCtor.has_value()) {
    firpContext()->beginContext(firpContext()->getClock(), firpContext()->getReset(), whenOp.getElseBodyBuilder());
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

Memory::Memory(FIRRTLBaseType dataType, size_t depth): dataType(dataType) {
  //size_t addrBits = clog2(depth - 1);

  TypeRange resultTypes{
    MemOp::getTypeForPort(depth, dataType, MemOp::PortKind::Write),
    MemOp::getTypeForPort(depth, dataType, MemOp::PortKind::Read)
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

FValue Memory::maskEnable() {
  FIRRTLBaseType maskType = dataType.getMaskType();

  std::function<FValue(FIRRTLBaseType)> constructOnes = [&](FIRRTLBaseType type) -> FValue {
    if (type.isGround()) {
      int32_t bitWidth = type.getBitWidthOrSentinel();
      assert(bitWidth >= 0);

      auto allOnes = llvm::APInt(bitWidth, 0, true) - 1;

      return firpContext()->builder().create<ConstantOp>(
        firpContext()->builder().getUnknownLoc(),
        uintType(bitWidth),
        allOnes
      ).getResult();
    }

    if (BundleType bundleType = llvm::dyn_cast<BundleType>(type)) {
      std::vector<Value> elementOnes;

      for (const auto& element : bundleType.getElements())
        elementOnes.push_back(constructOnes(element.type));

      return firpContext()->builder().create<BundleCreateOp>(
        firpContext()->builder().getUnknownLoc(),
        type,
        ArrayRef<Value>(elementOnes)
      ).getResult();
    }

    assert(false && "unsupported type");
    return Value();
  };

  return constructOnes(maskType);
}

void svVerbatim(const std::string& text) {
  firpContext()->builder().create<circt::sv::VerbatimOp>(
    firpContext()->builder().getUnknownLoc(),
    text
  );
}

}