#pragma once

#include <memory>
#include <stack>
#include <functional>

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


namespace llvm {

hash_code hash_value(const ::circt::firrtl::BundleType::BundleElement& element);
hash_code hash_value(const ::circt::firrtl::FIRRTLBaseType& type);

template <class T>
std::enable_if_t<std::negation_v<std::is_integral<T>>, hash_code> hash_value(const T&);

}

namespace firp {

using namespace ::mlir;
using namespace ::circt::firrtl;

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
public:
  DeclaredModules declaredModules;

  FirpContext(MLIRContext *ctxt, const std::string& topModule);
  FirpContext(ModuleOp root, const std::string& topModule);

  OpBuilder& builder() { return opBuilder; }
  MLIRContext *context() { return ctxt; }
  void dump() { root.dump(); }
  Value getClock() { return clock; }
  Value getReset() { return reset; }

  void beginContext(Value clock, Value reset, OpBuilder bodyBuilder);
  void endContext();

  void beginModuleDeclaration();
  void endModuleDeclaration();

  void finish();

  void verify() {
    assert(succeeded(::mlir::verify(root.getOperation(), true)));
  }
};

FirpContext *firpContext();
void initFirpContext(MLIRContext *mlirCtxt, const std::string& topModule);
void initFirpContext(ModuleOp root, const std::string& topModule);

// conventient type constructors

IntType uintType();
IntType uintType(uint32_t bitWidth);
IntType bitType();
ClockType clockType();
BundleType bundleType(std::initializer_list<std::tuple<std::string, bool, FIRRTLBaseType>> elements);
BundleType readyValidType(FIRRTLBaseType elementType);

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

  FValue operator()(size_t hi, size_t lo);
  FValue operator()(size_t index);
  FValue operator()(const std::string& fieldName);

  // Every FValue can be connected to any other FValue. However, subsequent
  // processing stages might fail if for example types or directions do not
  // fit together!
  void operator<<=(FValue other);

  FValue extend(size_t width);
};

FValue lift(Value val);
FValue cons(uint64_t n, IntType type = IntType::get(firpContext()->context(), false));
FValue mux(FValue cond, FValue pos, FValue neg);
FValue mux(FValue sel, std::initializer_list<FValue> options);
FValue zeros(FIRRTLBaseType type);
FValue ones(FIRRTLBaseType type);
FValue doesFire(FValue readyValidValue);

class Reg {
  RegResetOp regOp;
  FIRRTLBaseType type;
public:
  Reg(FIRRTLBaseType type, FValue resetValue, const std::string& name = "");
  Reg(FIRRTLBaseType type, const std::string& name = "");
  FValue read() { return regOp.getResult(); }
  void write(FValue what);
};

// mainly used for naming things and easier debugging
class Wire {
  WireOp wireOp;
  FIRRTLBaseType type;
public:
  Wire(FIRRTLBaseType type, const std::string& name = "");
  Wire(FValue what, const std::string& name = "");
  operator FValue();
  void operator<<=(FValue what);
};

struct Port {
  std::string name;
  bool isInput;
  FIRRTLBaseType type;

  Port() {}
  Port(const std::string& name, bool isInput, FIRRTLBaseType type):
    name(name), isInput(isInput), type(type) {}
};

template <class ConcreteModule>
class Module {
  llvm::hash_code hashValue;
  std::string baseName;
  std::string name;
  std::vector<Port> ports;
  std::unordered_map<std::string, uint32_t> portIndices;
  FModuleOp modOp;
  InstanceOp instOp;

  template <class...Args>
  void declare(Args&&...args) {
    std::vector<PortInfo> portInfos;
    for (const Port& port : ports)
      portInfos.emplace_back(
        firpContext()->builder().getStringAttr(port.name),
        port.type,
        port.isInput ? Direction::In : Direction::Out
      );

    firpContext()->beginModuleDeclaration();

    FModuleOp modOp = firpContext()->builder().create<FModuleOp>(
      firpContext()->builder().getUnknownLoc(),
      firpContext()->builder().getStringAttr(name),
      portInfos
    );

    firpContext()->declaredModules.addDeclared(hashValue, modOp);
    this->modOp = modOp;

    Value newClock = modOp.getBodyBlock()->getArguments()[0];
    Value newReset = modOp.getBodyBlock()->getArguments()[1];
    OpBuilder newBuilder = modOp.getBodyBuilder();

    firpContext()->beginContext(newClock, newReset, newBuilder);
    static_cast<ConcreteModule *>(this)->body(args...);
    firpContext()->endContext();

    firpContext()->endModuleDeclaration();
  }

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
      io("clock") <<= clk;

    if (rst)
      io("reset") <<= rst;
  }

  template <class...Args>
  static llvm::hash_code computeModuleHash(const std::string& name, const Args&...args) {
    auto codes = std::make_tuple(llvm::hash_value(args)...);

    return llvm::hash_combine(
      name,
      llvm::hash_value(codes)
    );
  }
public:
  // All Args must be hashable.
  template <class...Args>
  Module(const std::string& name, std::initializer_list<Port> ports, Args&&...args):
    hashValue(computeModuleHash(name, args...)),
    baseName(name),
    name(name + "_" + std::to_string(hashValue)),
    ports(ports) {

    // insert clock and reset port
    this->ports.insert(
      this->ports.begin(),
      Port("reset", true, bitType())
    );

    this->ports.insert(
      this->ports.begin(),
      Port("clock", true, ClockType::get(firpContext()->context()))
    );

    for (uint32_t i = 0; i < this->ports.size(); ++i)
      portIndices[this->ports[i].name] = i;

    if (!firpContext()->declaredModules.isDeclared(hashValue))
      declare(args...);

    modOp = firpContext()->declaredModules.getDeclared(hashValue);

    instantiate();
  }

  FValue io(const std::string& name) {
    // io() behaves differently depending on whether we are inside the currently
    // declared modules or are talking to the ports of an instance.

    // instOp != nullptr iff. it has already been declared!
    bool isFromOutside = instOp;

    if (isFromOutside)
      return instOp.getResults()[portIndices.at(name)];
    else
      return modOp.getBodyBlock()->getArguments()[portIndices.at(name)];
  }

  void makeTop() {
    instOp.getOperation()->erase();
    firpContext()->declaredModules.setTop(hashValue);
  }

  std::string getName() const {
    return name;
  }

  std::string getBaseName() const {
    return baseName;
  }
};

template <class ConcreteModule>
class ExternalModule {
  llvm::hash_code hashValue;
  std::string name;
  std::vector<Port> ports;
  std::unordered_map<std::string, uint32_t> portIndices;
  FExtModuleOp modOp;
  InstanceOp instOp;

  void declare() {
    std::vector<PortInfo> portInfos;
    for (const Port& port : ports)
      portInfos.emplace_back(
        firpContext()->builder().getStringAttr(port.name),
        port.type,
        port.isInput ? Direction::In : Direction::Out
      );

    firpContext()->beginModuleDeclaration();

    modOp = firpContext()->builder().create<FExtModuleOp>(
      firpContext()->builder().getUnknownLoc(),
      firpContext()->builder().getStringAttr(name),
      portInfos
    );

    firpContext()->declaredModules.addDeclared(hashValue, modOp);
    firpContext()->endModuleDeclaration();
  }

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
      io("clock") <<= clk;

    if (rst)
      io("reset") <<= rst;
  }
public:
  // All Args must be hashable.
  template <class Container>
  ExternalModule(const std::string& name, Container ports):
    hashValue(llvm::hash_value(name)),
    name(name),
    ports(std::cbegin(ports), std::cend(ports)) {

    // insert clock and reset port
    this->ports.insert(
      this->ports.begin(),
      Port("reset", true, bitType())
    );

    this->ports.insert(
      this->ports.begin(),
      Port("clock", true, ClockType::get(firpContext()->context()))
    );

    for (uint32_t i = 0; i < this->ports.size(); ++i)
      portIndices[this->ports[i].name] = i;

    if (!firpContext()->declaredModules.isDeclared(hashValue))
      declare();

    modOp = firpContext()->declaredModules.getExternalDeclared(hashValue);

    instantiate();
  }

  FValue io(const std::string& name) {
    // Connection always happens from outside with external modules.
    return instOp.getResults()[portIndices.at(name)];
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

BundleType memReadType(FIRRTLBaseType dataType, uint32_t addrBits);
BundleType memWriteType(FIRRTLBaseType dataType, uint32_t addrBits);

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
