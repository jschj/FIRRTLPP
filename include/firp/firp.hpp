#pragma once

#include <memory>
#include <stack>
#include <functional>
#include <unordered_map>

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/HWArith/HWArithTypes.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"

#include "llvm/ADT/DenseMap.h"


namespace firp {

template <class T>
std::enable_if_t<!std::is_integral_v<T>, llvm::hash_code> compute_hash(const T&);

template <class T>
std::enable_if_t<std::is_integral_v<T>, llvm::hash_code> compute_hash(const T& t) {
  return llvm::hash_value(t);
}

}

namespace llvm {

// essentially boils down to bit converting double to uint64_t
template<> struct DenseMapInfo<double> {
  static_assert(sizeof(double) == sizeof(uint64_t));

  static inline double getEmptyKey() {
    const uint64_t c = ~0ULL;
    return *reinterpret_cast<const double *>(&c);
  }

  static inline double getTombstoneKey() {
    const uint64_t c = ~0ULL - 1ULL;
    return *reinterpret_cast<const double *>(&c);
  }

  static unsigned getHashValue(const double& Val) {
    return (unsigned)(*reinterpret_cast<const uint64_t *>(&Val) * 37ULL);
  }

  static bool isEqual(const double &LHS, const double &RHS) {
    // standard floating point comparisons can get a little whacky
    return *reinterpret_cast<const uint64_t *>(&LHS) ==
      *reinterpret_cast<const uint64_t *>(&RHS);
  }
};

}

namespace firp {

using namespace ::mlir;
using namespace ::circt::firrtl;

/*
class DeclaredModules {
  llvm::DenseMap<llvm::hash_code, FModuleOp> declaredModules;
  std::optional<llvm::hash_code> topMod;
  llvm::DenseMap<llvm::hash_code, FExtModuleOp> externalDeclaredModules;
public:
  bool isDeclared(llvm::hash_code hashValue);
  void addDeclared(llvm::hash_code hashValue, FModuleOp decl);
  FModuleOp getDeclared(llvm::hash_code hashValue);
  void setTop(llvm::hash_code hashValue);
  FModuleOp getTop();

  void addDeclared(llvm::hash_code hashValue, FExtModuleOp decl);
  FExtModuleOp getExternalDeclared(llvm::hash_code hashValue);
};
 */

class ModuleBuilder;

// TODO: fix this mess
class FirpContext {
public:
  MLIRContext *ctxt;
  OpBuilder opBuilder;
  Value clock;
  Value reset;

  ModuleOp root;
  CircuitOp circuitOp;

  std::stack<OpBuilder> builderStack;
  std::stack<Value> clockStack;
  std::stack<Value> resetStack;

  std::string defaultClockName = "clock";
  std::string defaultResetName = "reset";
public:
  //DeclaredModules declaredModules;
  std::unique_ptr<ModuleBuilder> moduleBuilder;

  FirpContext(MLIRContext *ctxt):
    ctxt(ctxt), opBuilder(ctxt) {}

  //FirpContext(MLIRContext *ctxt, const std::string& topModule, const std::string& defaultClockName, const std::string& defaultResetName);
  //FirpContext(ModuleOp root, const std::string& topModule, const std::string& defaultClockName, const std::string& defaultResetName);
  //FirpContext(CircuitOp circuitOp, const std::string& defaultClockName, const std::string& defaultResetName);

  OpBuilder& builder() { return opBuilder; }
  MLIRContext *context() { return ctxt; }
  void dump() { root.dump(); }
  Value getClock() { return clock; }
  Value getReset() { return reset; }

  void beginContext(Value clock, Value reset, OpBuilder bodyBuilder);
  void endContext();

  void beginModuleDeclaration();
  void endModuleDeclaration();

  LogicalResult finish();

  void verify() {
    assert(succeeded(::mlir::verify(root.getOperation(), true)));
  }

  std::string getDefaultClockName() const { return defaultClockName; }
  std::string getDefaultResetName() const { return defaultResetName; }
};

FirpContext *firpContext();
//void initFirpContext(MLIRContext *mlirCtxt, const std::string& topModule, const std::string& defaultClockName = "clock", const std::string& defaultResetName = "reset");
//void initFirpContext(ModuleOp root, const std::string& topModule, const std::string& defaultClockName = "clock", const std::string& defaultResetName = "reset");
//void initFirpContext(CircuitOp circuitOp, const std::string& defaultClockName = "clock", const std::string& defaultResetName = "reset");

void createFirpContext(MLIRContext *ctxt, StringRef circuitName);
void attachFirpContext(ModuleOp modOp, StringRef circuitName);
void attachFirpContext(CircuitOp circuitOp);

class ModuleBuilder {
public:
  typedef std::function<void()> BodyCtor;
private:
  // normal modules
  struct Constructable {    
    FModuleOp modOp;
    BodyCtor bodyCtor;
  };

  llvm::DenseMap<uint32_t, Constructable> constructables;
  llvm::DenseSet<uint32_t> constructed;

  // external modules
  llvm::DenseMap<uint32_t, FExtModuleOp> extModules;

  // all the rest
  uint32_t inBodyOf = -1;
  uint32_t uid = 0;
  FModuleOp topOp;
public:
  void setInitialUid(uint32_t uid) {
    assert(this->uid == 0 && "uid already set");
    this->uid = uid;
  }

  // All args must implement DenseMapInfo. Use StringRef instead of std::string.
  template <class...Args>
  uint32_t getSignatureId(Args...args) {
    static llvm::DenseMap<std::tuple<Args...>, uint32_t> signatures;

    auto sig = std::make_tuple(args...);
    auto it = signatures.find(sig);

    if (it != signatures.end())
      return it->second;

    uint32_t id = uid++;
    signatures[sig] = id;

    return id;
  }

  // add a normal module
  template <class PortsT, class...Args>
  std::tuple<uint32_t, FModuleOp> addModule(PortsT ports, BodyCtor bodyCtor, StringRef baseName, Args...args) {
    // This function is instantiated for every unique combination of arg types.
    uint32_t sigId = getSignatureId(baseName, args...);

    // check if module has already been declared
    auto it = constructables.find(sigId);

    if (it != constructables.end())
      return std::make_tuple(it->first, it->second.modOp);

    std::string fullName = baseName.str() + "_" + std::to_string(sigId);

    // declare module
    std::vector<PortInfo> portInfos;
    for (const auto& port : ports)
      portInfos.emplace_back(
        firpContext()->builder().getStringAttr(port.name),
        port.type,
        port.isInput ? Direction::In : Direction::Out
      );
    
    firpContext()->beginModuleDeclaration();

    FModuleOp modOp = firpContext()->builder().create<FModuleOp>(
      firpContext()->builder().getUnknownLoc(),
      firpContext()->builder().getStringAttr(fullName),
      ConventionAttr::get(firpContext()->context(), Convention::Internal),
      portInfos
    );

    firpContext()->endModuleDeclaration();

    constructables[sigId] = Constructable{
      .modOp = modOp,
      .bodyCtor = bodyCtor
    };

    return std::make_tuple(sigId, modOp);
  }

  template <class PortsT>
  std::tuple<uint32_t, FExtModuleOp> addExternalModule(StringRef baseName, PortsT ports) {
    // An external module must have a signature and name that only depends on its base name.
    uint32_t sigId = getSignatureId(baseName);

    auto it = extModules.find(sigId);

    if (it != extModules.end())
      return std::make_tuple(it->first, it->second);

    std::vector<PortInfo> portInfos;
    for (const auto& port : ports)
      portInfos.emplace_back(
        firpContext()->builder().getStringAttr(port.name),
        port.type,
        port.isInput ? Direction::In : Direction::Out
      );

    firpContext()->beginModuleDeclaration();

    FExtModuleOp modOp = firpContext()->builder().create<FExtModuleOp>(
      firpContext()->builder().getUnknownLoc(),
      firpContext()->builder().getStringAttr(baseName),
      ConventionAttr::get(firpContext()->context(), Convention::Internal),
      portInfos
    );

    firpContext()->endModuleDeclaration();

    extModules[sigId] = modOp;

    return std::make_tuple(sigId, modOp);
  }

  void build(uint32_t sigId);

  bool isInBodyOf(uint32_t of) const {
    return of == inBodyOf;
  }

  void setTop(uint32_t sigId) {
    topOp = constructables[sigId].modOp;
  }

  FModuleOp getTop() {
    return topOp;
  }

  bool hasUnfinishedConstructions() const {
    return constructed.size() < constructables.size();
  }
};

// convenient type functions

IntType uintType();
UIntType uintType(uint32_t bitWidth);
SIntType sintType(uint32_t bitWidth);
IntType bitType();
ClockType clockType();

template <class Container = std::initializer_list<std::tuple<std::string, bool, FIRRTLBaseType>>>
BundleType bundleType(Container elements) {
  std::vector<BundleType::BundleElement> els;

  for (const auto& [name, flip, type] : elements)
    els.push_back(BundleType::BundleElement(
      firpContext()->builder().getStringAttr(name), flip, type
    ));

  return BundleType::get(firpContext()->context(), els);
}

BundleType readyValidType(FIRRTLBaseType elementType);
FIRRTLBaseType flattenType(FIRRTLBaseType type, const std::string& infix);
FVectorType vectorType(FIRRTLBaseType elementType, uint32_t n);

class ConnectResult {
  bool successful;
  std::string msg;
public:
  ConnectResult(): successful(true) {}
  ConnectResult(const std::string& msg): successful(false), msg(msg) {}
  std::string getMessage() const { return msg; }
  operator bool() const { return successful; }
  static ConnectResult success() { return ConnectResult(); }
  static ConnectResult failure(const std::string& msg) { return ConnectResult(msg); }
};

class FValue : public Value {
public:
  FValue() {}
  FValue(Value val): Value(val) {}

  FValue operator~();

  FValue operator+(FValue other);
  FValue operator-(FValue other);
  FValue operator*(FValue other);
  FValue operator/(FValue other);
  FValue operator&(FValue other);
  FValue operator|(FValue other);
  FValue operator>(FValue other);
  FValue operator>=(FValue other);
  FValue operator<(FValue other);
  FValue operator<=(FValue other);
  FValue operator==(FValue other);
  FValue operator!=(FValue other);

  // bit extraction
  FValue operator()(size_t hi, size_t lo);
  FValue operator()(size_t index);
  // struct field extraction
  FValue operator()(const std::string& fieldName);
  // vector extraction (dynamic and static)
  FValue operator[](FValue index);
  FValue operator[](size_t index);

  FValue operator<<(uint32_t amount);
  FValue operator>>(uint32_t amount);
  FValue operator<<(FValue amount);
  FValue operator>>(FValue amount);

  // Every FValue can be connected to any other FValue. However, subsequent
  // processing stages might fail if for example types or directions do not
  // fit together!
  ConnectResult operator<<=(FValue other);

  FValue extend(size_t width);

  uint32_t bitCount();
  // extract the most significant n bits
  FValue head(uint32_t n);
  // drop the most significant n bits
  FValue tail(uint32_t n);
  FValue asSInt();
  FValue asUInt();
  // horizontally fold bits with or operation
  FValue orr();
};

FValue lift(Value val);
FValue cons(uint64_t n, IntType type = IntType::get(firpContext()->context(), false));
FValue uval(uint64_t n, int32_t bitCount = -1);
FValue sval(int64_t n, int32_t bitCount = -1);
FValue mux(FValue cond, FValue pos, FValue neg);
FValue mux(FValue sel, std::initializer_list<FValue> options);
FValue zeros(FIRRTLBaseType type);
FValue ones(FIRRTLBaseType type);
FValue doesFire(FValue readyValidValue);
FValue clockToInt(FValue clock);
FValue shiftRegister(FValue input, uint32_t delay);

Value hwStructCast(Value input, Type targetType);

//inline FValue operator""_u(unsigned long long n) {
//  return uint(n);
//}

template <class Container = std::initializer_list<FValue>>
FValue cat(Container values) {
  assert(values.size() > 0 && "cat is not defined for 0 arguments");

  FValue lhs = *values.begin();

  // skip the first one
  for (auto it = values.begin() + 1; it != values.end(); ++it)
    lhs = firpContext()->builder().create<CatPrimOp>(
      firpContext()->builder().getUnknownLoc(),
      lhs, *it
    ).getResult();

  return lhs;  
}

template <class Container = std::initializer_list<FValue>>
FValue vector(Container values) {
  std::vector<Value> vec(std::begin(values), std::end(values));
  assert(vec.size() >= 1 && "cannot create empty vector");

  Type firstType = vec.front().getType();

  for (const auto& el : vec)
    assert(el.getType() == firstType && "types must match");

  auto resultType = FVectorType::get(firstType.dyn_cast<FIRRTLBaseType>(), vec.size());

  return firpContext()->builder().create<VectorCreateOp>(
    firpContext()->builder().getUnknownLoc(),
    resultType,
    vec    
  ).getResult();
}

template <class Container = std::initializer_list<FValue>>
FValue bundleCreate(BundleType type, Container container) {
  std::vector<Value> values;

  for (auto e : container)
    values.push_back(e);

  return firpContext()->builder().create<BundleCreateOp>(
    firpContext()->builder().getUnknownLoc(),
    type,
    ArrayRef<Value>(values)
  ).getResult();
}

template <class Iterator>
FValue treeFold(Iterator begin, Iterator end, const std::function<FValue(FValue, FValue)>& associativeOp) {
  size_t size = std::distance(begin, end);
  
  assert(size > 0 && "treeFold is not defined for 0 arguments");

  if (size == 1)
    return *begin;

  if (size == 2)
    return associativeOp(*begin, *(begin + 1));

  size_t half = size / 2;

  FValue lhs = treeFold(begin, begin + half, associativeOp);
  FValue rhs = treeFold(begin + half, end, associativeOp);

  return associativeOp(lhs, rhs);
}

//constexpr FValue operator""_u(unsigned long long int n) {
//  return FValue();
//}

class Reg {
  RegResetOp regOp;
  FIRRTLBaseType type;
public:
  Reg(FIRRTLBaseType type, FValue resetValue, const std::string& name = "");
  Reg(FIRRTLBaseType type, const std::string& name = "");
  // explicit read/write interfaces for cases where automatic type conversion fails
  FValue read() { return regOp.getResult(); }
  void write(FValue what) { FValue(regOp.getResult()) <<= what; }
  operator FValue() { return read(); }
  void operator<<=(FValue what) { write(what); }
  FValue operator()(const std::string& fieldName) { return read()(fieldName); }
  FValue operator()(uint32_t i) { return read()(i); }
  FValue operator()(uint32_t hi, uint32_t lo) { return read()(hi, lo); }
};

// mainly used for naming things to make debugging with GTKWave easier
class Wire {
  WireOp wireOp;
  FIRRTLBaseType type;
public:
  Wire(FIRRTLBaseType type, const std::string& name = "");
  // explicit read/write interfaces for cases where automatic type conversion fails
  FValue read() { return wireOp.getResult(); }
  void write(FValue what) { FValue(wireOp.getResult()) <<= what; }
  operator FValue() { return read(); }
  void operator<<=(FValue what) { write(what); }
  FValue operator()(const std::string& fieldName) { return read()(fieldName); }
  FValue operator()(uint32_t i) { return read()(i); }
  FValue operator()(uint32_t hi, uint32_t lo) { return read()(hi, lo); }
};

// Chisel-like convenience functions
Reg regNext(FValue what, const std::string& name = "");
Reg regInit(FValue init, const std::string& name = "");
Reg regNextWhen(FValue what, FValue cond, const std::string& name = "");
Wire wireInit(FValue what, const std::string& name = "");
FValue named(FValue what, const std::string& name);

struct Port {
  std::string name;
  bool isInput;
  Type type;

  Port() {}
  Port(const std::string& name, bool isInput, Type type):
    name(name), isInput(isInput), type(type) {}
};

Port Input(const std::string& name, Type type);
Port Output(const std::string& name, Type type);

template <class ConcreteModule>
class Module {
  std::string baseName;
  std::string name;
  std::vector<Port> ports;
  std::unordered_map<std::string, uint32_t> portIndices;
  uint32_t signatureId;
  FModuleOp modOp;
  InstanceOp instOp;
  bool wasBuilt = false;

  void instantiate() {
    instOp = firpContext()->builder().create<InstanceOp>(
      firpContext()->builder().getUnknownLoc(),
      modOp,
      firpContext()->builder().getStringAttr(name + "_instance")
    );

    // instantiating includes connecting clk and rst
    Value clk = firpContext()->getClock();
    Value rst = firpContext()->getReset();

    // This check is necessary because a module is ALWAYS instantiated, even
    // the top one. The instantiation is only removed afterwards.
    if (clk)
      io(firpContext()->getDefaultClockName()) <<= clk;

    if (rst)
      io(firpContext()->getDefaultResetName()) <<= rst;
  }
public:
  // All Args must be hashable.
  template <class Container = std::initializer_list<Port>, class...Args>
  Module(const std::string& name, Container ports, Args&&...args):
    baseName(name),
    ports(ports) {

    // insert clock and reset port
    this->ports.insert(
      this->ports.begin(),
      Port(firpContext()->getDefaultResetName(), true, bitType())
    );

    this->ports.insert(
      this->ports.begin(),
      Port(firpContext()->getDefaultClockName(), true, ClockType::get(firpContext()->context()))
    );

    for (uint32_t i = 0; i < this->ports.size(); ++i)
      portIndices[this->ports[i].name] = i;

    auto result = firpContext()->moduleBuilder->addModule(this->ports,
      std::bind(&ConcreteModule::body, static_cast<ConcreteModule *>(this)),
      StringRef(name), args...);

    signatureId = std::get<0>(result);
    modOp = std::get<1>(result);

    this->name = baseName + "_" + std::to_string(signatureId);

    instantiate();
  }

  virtual ~Module() {
    // Doing construction here retains the natural hierarchy. Additionally, it ensures
    // that the bound this pointer of the body constructor is still valid.
    //firpContext()->moduleBuilder->build(signatureId);

    //build();

    assert(wasBuilt && "Module was not built. Did you call build() in the constructor?");
  }

  void build() {
    // BUG: ???

    // check that build() was not already called for this module
    assert(!wasBuilt && "Module was already built. Make sure that build() is only called once.");
    firpContext()->moduleBuilder->build(signatureId);
    wasBuilt = true;
  }

  FValue io(const std::string& name) {
    // io() behaves differently depending on whether we are inside the currently
    // declared modules or are talking to the ports of an instance.

    auto it = portIndices.find(name);

    if (it == portIndices.end()) {
      llvm::errs() << "port " << name << " does not exist\n";
      assert(false && "port not found");
    }

    uint32_t index = it->second;

    bool isFromOutside = !firpContext()->moduleBuilder->isInBodyOf(signatureId);

    if (isFromOutside)
      return instOp.getResults()[index];
    else
      return modOp.getBodyBlock()->getArguments()[index];
  }

  FModuleOp makeTop() {
    //removeInstance();

    //firpContext()->moduleBuilder->setTop(signatureId);

    // build a wrapper module with the base name
    firpContext()->beginModuleDeclaration();

    FModuleOp wrapperOp = firpContext()->builder().create<FModuleOp>(
      firpContext()->builder().getUnknownLoc(),
      firpContext()->builder().getStringAttr(baseName),
      ConventionAttr::get(firpContext()->context(), Convention::Internal),
      modOp.getPorts()
    );
    llvm::outs() << "created wrapper module for " << baseName << "\n";

    Value newClock = wrapperOp.getBodyBlock()->getArgument(0);
    Value newReset = wrapperOp.getBodyBlock()->getArgument(1);
    firpContext()->beginContext(newClock, newReset, wrapperOp.getBodyBuilder());
    llvm::outs() << "began new context\n";

    InstanceOp cloned = cast<InstanceOp>(instOp->clone());
    instOp->erase();
    firpContext()->builder().insert(cloned);
    //instOp->erase();

    for (size_t i = 0; i < wrapperOp.getNumPorts(); ++i) {
      FValue arg = wrapperOp.getArguments()[i];
      FValue res = cloned.getResults()[i];

      if (modOp.getPorts()[i].direction == Direction::In)
        res <<= arg;
      else
        arg <<= res;
    }

    firpContext()->endContext();
    firpContext()->endModuleDeclaration();

    return wrapperOp;
  }

  void removeInstance() {
    instOp.getOperation()->erase();
  }

  std::string getName() const {
    return name;
  }

  std::string getBaseName() const {
    return baseName;
  }

  FModuleOp getModuleOp() const {
    return modOp;
  }

  uint32_t getPortIndex(const std::string& name) const {
    auto it = portIndices.find(name);

    if (it == portIndices.end()) {
      llvm::errs() << "port " << name << " does not exist\n";
      assert(false && "port not found");
    }

    return it->second;
  }
};

template <class ConcreteModule>
class ExternalModule {
  std::string name;
  std::vector<Port> ports;
  std::unordered_map<std::string, uint32_t> portIndices;
  uint32_t signatureId;
  FExtModuleOp modOp;
  InstanceOp instOp;

  void instantiate() {
    instOp = firpContext()->builder().create<InstanceOp>(
      firpContext()->builder().getUnknownLoc(),
      modOp,
      firpContext()->builder().getStringAttr(name + "_instance")
    );

    // instantiating includes connecting clk and rst
    Value clk = firpContext()->getClock();
    Value rst = firpContext()->getReset();

    // This check is necessary because a module is ALWAYS instantiated, even
    // the top one. The instantiation is only removed afterwards.
    if (clk)
      io(firpContext()->getDefaultClockName()) <<= clk;

    if (rst)
      io(firpContext()->getDefaultResetName()) <<= rst;
  }
public:
  // All Args must be hashable.
  template <class Container = std::initializer_list<Port>>
  ExternalModule(const std::string& name, Container ports):
    name(name),
    ports(std::cbegin(ports), std::cend(ports)) {

    // insert clock and reset port
    this->ports.insert(
      this->ports.begin(),
      Port(firpContext()->getDefaultResetName(), true, bitType())
    );

    this->ports.insert(
      this->ports.begin(),
      Port(firpContext()->getDefaultClockName(), true, ClockType::get(firpContext()->context()))
    );

    for (uint32_t i = 0; i < this->ports.size(); ++i)
      portIndices[this->ports[i].name] = i;

    auto result = firpContext()->moduleBuilder->addExternalModule(StringRef(name), this->ports);
    signatureId = std::get<0>(result);
    modOp = std::get<1>(result);

    instantiate();
  }

  FValue io(const std::string& name) {
    // Connection always happens from outside with external modules.
    auto it = portIndices.find(name);

    if (it == portIndices.end()) {
      llvm::errs() << "port " << name << " does not exist\n";
      assert(false && "port not found");
    }

    uint32_t index = it->second;
    return instOp.getResults()[index];
  }

  StringRef getInstanceName() {
    return instOp.getName();
  }
};

class FIRModule {
  FModuleOp modOp;
  InstanceOp instOp;
  std::unordered_map<std::string, uint32_t> portIndices;
public:
  FIRModule(FModuleOp modOp) {
    this->modOp = modOp;
    this->instOp = firpContext()->builder().create<InstanceOp>(
      firpContext()->builder().getUnknownLoc(),
      modOp,
      firpContext()->builder().getStringAttr(modOp.getName().str() + "_instance")
    );

    uint32_t index = 0;
    for (const PortInfo& portInfo: modOp.getPorts())
      portIndices[portInfo.getName().str()] = index++;

    llvm::outs() << "instantiated " << modOp.getName() << " with the following ports:\n";
    for (const PortInfo& portInfo: modOp.getPorts())
      llvm::outs() << "  " << portInfo.getName() << "\n";
  }

  FValue io(const std::string& name) {
    auto it = portIndices.find(name);

    if (it == portIndices.end()) {
      llvm::errs() << "port " << name << " does not exist\n";
      assert(false && "port not found");
    }

    uint32_t index = it->second;
    return instOp.getResults()[index];
  }
};

class Conditional {
public:
  typedef std::function<void()> BodyCtor;
private:
  std::vector<std::tuple<FValue, BodyCtor>> cases;
  Optional<BodyCtor> otherwiseCtor;

  void build(size_t i, OpBuilder builder);
public:
  Conditional(FValue cond, BodyCtor bodyCtor);
  Conditional& elseWhen(FValue cond, BodyCtor bodyCtor);
  Conditional& otherwise(BodyCtor bodyCtor);
  ~Conditional();
};

Conditional when(FValue cond, Conditional::BodyCtor bodyCtor);

template <class ScalarType>
inline std::enable_if_t<std::is_scalar_v<ScalarType>, ScalarType> clog2(const ScalarType& value) {
  ScalarType n(1);
  ScalarType bits(1);

  while (n * 2 - 1 < value) {
    ++bits;
    n *= 2;
  }

  return bits;
}

class Memory {
  MemOp memOp;
  FIRRTLBaseType dataType;
public:
  Memory(FIRRTLBaseType dataType, size_t depth);
  FValue writePort();
  FValue readPort();
  FValue maskEnable();
};

void svVerbatim(const std::string& text);
void svCocoTBVerbatim(const std::string& moduleName);

}
