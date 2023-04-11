#include "firp.hpp"


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

bool DeclaredModules::isDeclared(llvm::hash_code hashValue) {
  return declaredModules.find(hashValue) != declaredModules.end() ||
    externalDeclaredModules.find(hashValue) != externalDeclaredModules.end();
}

void DeclaredModules::addDeclared(llvm::hash_code hashValue, FModuleOp decl) {
  declaredModules[hashValue] = decl;
}

FModuleOp DeclaredModules::getDeclared(llvm::hash_code hashValue) {
  return declaredModules[hashValue];
}

void DeclaredModules::setTop(llvm::hash_code hashValue) {
  assert(isDeclared(hashValue));
  topMod = hashValue;
}

FModuleOp DeclaredModules::getTop() {
  assert(topMod.has_value());
  return declaredModules[topMod.value()];
}

void DeclaredModules::addDeclared(llvm::hash_code hashValue, FExtModuleOp decl) {
  externalDeclaredModules[hashValue] = decl;
}

FExtModuleOp DeclaredModules::getExternalDeclared(llvm::hash_code hashValue) {
  return externalDeclaredModules[hashValue];
}

FirpContext::FirpContext(MLIRContext *ctxt, const std::string& topModule): ctxt(ctxt), opBuilder(ctxt) {
  root = builder().create<ModuleOp>(
    builder().getUnknownLoc()
  );

  builder().setInsertionPointToStart(
    &root.getBodyRegion().front()
  );

  return;

  circuitOp = builder().create<CircuitOp>(
    builder().getUnknownLoc(),
    builder().getStringAttr(topModule)
  );

  opBuilder = circuitOp.getBodyBuilder();
}

FirpContext::FirpContext(ModuleOp root, const std::string& topModule): ctxt(root.getContext()), opBuilder(ctxt) {
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

void FirpContext::finish() {
  // Our top module currently has the name "MyTop_<some hash value>" whereas CircuitOp
  // is called MyTop. We construct a module named MyTop that wraps MyTop_<some hash value>.

  beginModuleDeclaration();

  FModuleOp top = declaredModules.getTop();
  FModuleOp wrapper = opBuilder.create<FModuleOp>(
    opBuilder.getUnknownLoc(),
    opBuilder.getStringAttr(circuitOp.getName()),
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
}

static std::unique_ptr<FirpContext> ctxt;

FirpContext *firpContext() {
  return ctxt.get();
}

void initFirpContext(MLIRContext *mlirCtxt, const std::string& topModule) {
  ctxt = std::make_unique<FirpContext>(mlirCtxt, topModule);
}

void initFirpContext(ModuleOp root, const std::string& topModule) {
  ctxt = std::make_unique<FirpContext>(root, topModule);
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

FValue doesFire(FValue readyValidValue) {
  return readyValidValue("valid") & readyValidValue("ready");
}

FValue clockToInt(FValue clock) {
  return firpContext()->builder().create<AsUIntPrimOp>(
    firpContext()->builder().getUnknownLoc(),
    clock
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
    what.getType(),
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

IntType uintType(uint32_t bitWidth) {
  return IntType::get(firpContext()->context(), false, bitWidth);
}

IntType bitType() {
  return uintType(1);
}

ClockType clockType() {
  return ClockType::get(firpContext()->context());
}

BundleType bundleType(std::initializer_list<std::tuple<std::string, bool, FIRRTLBaseType>> elements) {
  std::vector<BundleType::BundleElement> els;
  for (const auto& [name, flip, type] : elements)
    els.push_back(BundleType::BundleElement(
      firpContext()->builder().getStringAttr(name), flip, type
    ));

  return BundleType::get(firpContext()->context(), els);
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