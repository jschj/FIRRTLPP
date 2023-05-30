#include <firp/firp.hpp>


namespace firp {

template <>
llvm::hash_code compute_hash(const std::string& s) {
  return llvm::hash_value(s);
}

template <>
llvm::hash_code compute_hash(const circt::firrtl::FIRRTLBaseType& t) {
  return mlir::hash_value(t);
}

}

namespace firp {

void ModuleBuilder::build(uint32_t sigId) {
  auto constructable = constructables[sigId];
  FModuleOp modOp = constructable.modOp;

  Value newClock = modOp.getBodyBlock()->getArguments()[0];
  Value newReset = modOp.getBodyBlock()->getArguments()[1];
  OpBuilder newBuilder = modOp.getBodyBuilder();

  firpContext()->beginContext(newClock, newReset, newBuilder);
  uint32_t oldBody = inBodyOf;
  inBodyOf = sigId;
  constructable.bodyCtor();
  inBodyOf = oldBody;
  firpContext()->endContext();

  constructed.insert(sigId);
}

FirpContext::FirpContext(MLIRContext *ctxt, const std::string& topModule, const std::string& defaultClockName, const std::string& defaultResetName):
  ctxt(ctxt), opBuilder(ctxt), defaultClockName(defaultClockName), defaultResetName(defaultResetName),
  moduleBuilder(std::make_unique<ModuleBuilder>()) {
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

FirpContext::FirpContext(ModuleOp root, const std::string& topModule, const std::string& defaultClockName, const std::string& defaultResetName):
  ctxt(root.getContext()), opBuilder(ctxt), defaultClockName(defaultClockName), defaultResetName(defaultResetName),
  moduleBuilder(std::make_unique<ModuleBuilder>()) {
  this->root = root;

  builder().setInsertionPointToStart(
    &root.getBodyRegion().back()
  );

  circuitOp = builder().create<CircuitOp>(
    builder().getUnknownLoc(),
    builder().getStringAttr(topModule)
  );

  opBuilder = circuitOp.getBodyBuilder();
}

FirpContext::FirpContext(CircuitOp circuitOp, const std::string& defaultClockName, const std::string& defaultResetName):
  opBuilder(ctxt) {
  this->ctxt = circuitOp.getContext();
  this->circuitOp = circuitOp;
  this->defaultClockName = defaultClockName;
  this->defaultResetName = defaultResetName;
  this->root = circuitOp->getParentOfType<ModuleOp>();
  this->moduleBuilder = std::make_unique<ModuleBuilder>();

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

FModuleOp FirpContext::finish() {
  assert(!moduleBuilder->hasUnfinishedConstructions() && "Some modules have not been constructed. Make sure their destructor is called before calling finish().");

  // Our top module currently has the name "MyTop_<some hash value>" whereas CircuitOp
  // is called MyTop. We construct a module named MyTop that wraps MyTop_<some hash value>.

  beginModuleDeclaration();

  FModuleOp top = moduleBuilder->getTop();
  assert(top && "Top module not defined. Use makeTop() to set your top module.");

  FModuleOp wrapper = opBuilder.create<FModuleOp>(
    opBuilder.getUnknownLoc(),
    opBuilder.getStringAttr(circuitOp.getName()),
    ConventionAttr::get(firpContext()->context(), Convention::Internal),
    top.getPorts()
  );

  beginContext(getClock(), getReset(), wrapper.getBodyBuilder());

  InstanceOp inst = opBuilder.create<InstanceOp>(
    opBuilder.getUnknownLoc(),
    top,
    "wrapper_instance"
  );

  for (size_t i = 0; i < top.getNumPorts(); ++i) {
    FValue arg = wrapper.getArguments()[i];
    FValue res = inst.getResults()[i];

    if (top.getPorts()[i].direction == Direction::In)
      res <<= arg;
    else
      arg <<= res;
  }

  endContext();

  endModuleDeclaration();

  return wrapper;
}

static std::unique_ptr<FirpContext> ctxt;

FirpContext *firpContext() {
  return ctxt.get();
}

void initFirpContext(MLIRContext *mlirCtxt, const std::string& topModule, const std::string& defaultClockName, const std::string& defaultResetName) {
  ctxt = std::make_unique<FirpContext>(mlirCtxt, topModule, defaultClockName, defaultResetName);
}

void initFirpContext(ModuleOp root, const std::string& topModule, const std::string& defaultClockName, const std::string& defaultResetName) {
  ctxt = std::make_unique<FirpContext>(root, topModule, defaultClockName, defaultResetName);
}

void initFirpContext(CircuitOp circuitOp, const std::string& defaultClockName, const std::string& defaultResetName) {
  ctxt = std::make_unique<FirpContext>(circuitOp, defaultClockName, defaultResetName);
}

FValue lift(Value val) {
  return FValue(val);
}

FValue cons(uint64_t n, IntType type) {
  ::llvm::APInt concreteValue =
    type.getWidthOrSentinel() == -1 ?
    ::llvm::APInt(64, n) :
    ::llvm::APInt(type.getWidthOrSentinel(), n);

  FValue val = ctxt->builder().create<ConstantOp>(
    ctxt->builder().getUnknownLoc(),
    type,
    concreteValue
  ).getResult();

  return val;
}

FValue uval(uint64_t n, int32_t bitCount) {
  if (bitCount == -1)
    bitCount = clog2(n);
  else
    assert(clog2(n) <= uint64_t(bitCount) && "Value does not fit in bitCount");

  llvm::APInt value(bitCount, n, false);

  return ctxt->builder().create<ConstantOp>(
    ctxt->builder().getUnknownLoc(),
    uintType(bitCount),
    value
  ).getResult();
}

FValue sval(int64_t n, int32_t bitCount) {
  if (bitCount == -1)
    bitCount = clog2(std::abs(n)) + 1;
  else
    assert(clog2(std::abs(n)) + 1 <= int64_t(bitCount) && "Value does not fit in bitCount");

  llvm::APInt value(bitCount, n, true);

  return ctxt->builder().create<ConstantOp>(
    ctxt->builder().getUnknownLoc(),
    sintType(bitCount),
    value
  ).getResult();
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
  if (BundleType bundleType = type.dyn_cast<BundleType>()) {
    SmallVector<FValue> zeroValues;

    for (auto el : bundleType.getElements())
      zeroValues.push_back(zeros(el.type));

    return bundleCreate(bundleType, zeroValues);
  } else if (IntType intType = type.dyn_cast<IntType>()) {
    int32_t width = intType.getBitWidthOrSentinel();
    assert(width >= 0 && "cannot create zeroes for int type with uninferred width");
    return cons(0, intType);
  } else {
    assert(false && "type not supported");
  }
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

FValue doesFire(FValue readyValidValue) {
  return readyValidValue("valid") & readyValidValue("ready");
}

FValue clockToInt(FValue clock) {
  return firpContext()->builder().create<AsUIntPrimOp>(
    firpContext()->builder().getUnknownLoc(),
    clock
  ).getResult();
}

FValue shiftRegister(FValue input, uint32_t delay) {
  FValue result = input;

  for (uint32_t i = 0; i < delay; ++i)
    result = regNext(result);

  return result;
}

FValue FValue::operator~() {
  return firpContext()->builder().create<NotPrimOp>(firpContext()->builder().getUnknownLoc(), *this).getResult();
}

FValue FValue::operator+(FValue other) {
  return firpContext()->builder().create<AddPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator-(FValue other) {
  return firpContext()->builder().create<SubPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator*(FValue other) {
  return firpContext()->builder().create<MulPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator/(FValue other) {
  return firpContext()->builder().create<DivPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator&(FValue other) {
  return firpContext()->builder().create<AndPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator|(FValue other) {
  return firpContext()->builder().create<OrPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator>(FValue other) {
  return firpContext()->builder().create<GTPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator>=(FValue other) {
  return firpContext()->builder().create<GEQPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator<(FValue other) {
  return firpContext()->builder().create<LTPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator<=(FValue other) {
  return firpContext()->builder().create<LEQPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator==(FValue other) {
  return firpContext()->builder().create<EQPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
}

FValue FValue::operator!=(FValue other) {
  return firpContext()->builder().create<NEQPrimOp>(firpContext()->builder().getUnknownLoc(), *this, other).getResult();
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

FValue FValue::operator[](FValue index) {
  return firpContext()->builder().create<SubaccessOp>(
    firpContext()->builder().getUnknownLoc(),
    *this,
    index
  ).getResult();
}

FValue FValue::operator<<(uint32_t amount) {
  return firpContext()->builder().create<ShlPrimOp>(
    firpContext()->builder().getUnknownLoc(),
    *this,
    amount
  ).getResult();
}

FValue FValue::operator>>(uint32_t amount) {
  return firpContext()->builder().create<ShrPrimOp>(
    firpContext()->builder().getUnknownLoc(),
    *this,
    amount
  ).getResult();
}

FValue FValue::operator<<(FValue amount) {
  FValue result = *this;

  for (uint32_t i = 0; i < amount.bitCount(); ++i) {
    uint32_t amt = 1 << i;
    result = mux(amount(i), result << amt, result);
  }

  return result;
}

FValue FValue::operator>>(FValue amount) {
  FValue result = *this;

  for (uint32_t i = 0; i < amount.bitCount(); ++i) {
    uint32_t amt = 1 << i;
    result = mux(amount(i), result >> amt, result);
  }

  return result;
}

ConnectResult FValue::operator<<=(FValue other) {
  // We use ConnectOp because it is more forgiving. Additionally, (in contrast to the spec)
  // we allow bit widths to be truncated (use case: reg <<= reg + cons(1)).
  // We make use of ConnectOp's verify() function to confirm the we did everything right.
  // If not, we return some diagnostics to help the user.
  // Our connection operator is designed to catch common mistakes early and be convenient,
  // but should not be considered truly faithful to the spec!
  FValue src, dst;

  src = other;
  dst = *this;

  if (llvm::isa<IntType>(getType()) && llvm::isa<IntType>(other.getType())) {
    IntType dstInt = llvm::dyn_cast<IntType>(getType());
    IntType srcInt = llvm::dyn_cast<IntType>(other.getType());

    int32_t dstWidth = dstInt.getBitWidthOrSentinel();
    int32_t srcWidth = srcInt.getBitWidthOrSentinel();

    // Truncate!
    if (dstWidth >= 0 && srcWidth > dstWidth)
      src = other(dstWidth - 1, 0);
  }

  ConnectOp connectOp = firpContext()->builder().create<ConnectOp>(
    firpContext()->builder().getUnknownLoc(),
    dst,
    src
  );

  if (succeeded(connectOp.verify()))
    return ConnectResult::success();
  else
    return ConnectResult::failure("Could not connect values! Please check your code and the spec.");
}

FValue FValue::extend(size_t width) {
  assert(llvm::isa<IntType>(getType()));

  return firpContext()->builder().create<PadPrimOp>(
    firpContext()->builder().getUnknownLoc(),
    *this,
    firpContext()->builder().getI32IntegerAttr(width)
  ).getResult();
}

uint32_t FValue::bitCount() {
  if (IntType intType = llvm::dyn_cast<IntType>(getType()))
    if (intType.getBitWidthOrSentinel() != -1)
      return uint32_t(intType.getBitWidthOrSentinel());

  throw std::runtime_error("Type is not an int type or has unknown width!");
}

FValue FValue::head(uint32_t n) {
  uint32_t m = bitCount();
  return (*this)(m - 1, m - n);
}

FValue FValue::tail(uint32_t n) {
  uint32_t m = bitCount();
  return (*this)(m - n - 1, 0);
}

FValue FValue::asSInt() {
  return firpContext()->builder().create<AsSIntPrimOp>(
    firpContext()->builder().getUnknownLoc(),
    *this
  ).getResult();
}

FValue FValue::asUInt() {
  return firpContext()->builder().create<AsUIntPrimOp>(
    firpContext()->builder().getUnknownLoc(),
    *this
  ).getResult();
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

Wire::Wire(FIRRTLBaseType type, const std::string& name) {
  wireOp = firpContext()->builder().create<WireOp>(
    firpContext()->builder().getUnknownLoc(),
    type,
    name
  );
  this->type = type;
}

Reg regNext(FValue what, const std::string& name) {
  Reg reg(llvm::dyn_cast<FIRRTLBaseType>(what.getType()), name);
  reg <<= what;
  return reg;
}

Reg regInit(FValue init, const std::string& name) {
  Reg reg(llvm::dyn_cast<FIRRTLBaseType>(init.getType()), init, name);
  return reg;
}

Wire wireInit(FValue what, const std::string& name) {
  Wire wire(llvm::dyn_cast<FIRRTLBaseType>(what.getType()), name);
  wire <<= what;
  return wire;
}

FValue named(FValue what, const std::string& name) {
  return firpContext()->builder().create<NodeOp>(
    firpContext()->builder().getUnknownLoc(),
    //what.getType(),
    what,
    firpContext()->builder().getStringAttr(name)
  ).getResult();
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

UIntType uintType(uint32_t bitWidth) {
  return UIntType::get(firpContext()->context(), bitWidth);
}

SIntType sintType(uint32_t bitWidth) {
  return SIntType::get(firpContext()->context(), bitWidth);
}

IntType bitType() {
  return uintType(1);
}

ClockType clockType() {
  return ClockType::get(firpContext()->context());
}

BundleType readyValidType(FIRRTLBaseType elementType) {
  return bundleType({
    {"ready", true, bitType()},
    {"valid", false, bitType()},
    {"bits", false, elementType}
  });
}

/*
FIRRTLBaseType flattenType(FIRRTLBaseType type, const std::string& infix) {  
  BundleType bundleType = llvm::dyn_cast<BundleType>(type);

  if (!bundleType)
    return type;

  std::vector<BundleType::BundleElement> newElements;

  for (const auto& el : bundleType.getElements()) {
    FIRRTLBaseType flattened = flattenType(el.type);
    BundleType bundleType = llvm::dyn_cast<BundleType>(flattened);

    if (!bundleType) {
      newElements.emplace_back(el.name, el.isFlip, flattened);
      continue;
    }

    for (const auto& flattenedEl : flattened.getElements())
      newElements.emplace_back(el.name + infix + flattenedEl.name, flattenedEl.isFlip, flattenedEl.type);
  }

  return bundleType(newElements);
}*/

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

void svCocoTBVerbatim(const std::string& moduleName) {
  std::stringstream ss;
  ss << "`ifdef COCOTB_SIM\n"
     << "  initial begin\n"
     << "    $dumpfile(\"" << moduleName << ".vcd\");\n"
     << "    $dumpvars (0, " << moduleName << ");\n"
     << "    #1;\n"
     << "  end\n"
     << "`endif\n";

  svVerbatim(ss.str());
}

}