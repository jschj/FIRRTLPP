#pragma once

#include <memory>
#include <stack>

#include "mlir/IR/PatternMatch.h"
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

namespace firp {

using namespace ::mlir;
using namespace ::circt::firrtl;

class FirpContext {
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
  FirpContext(MLIRContext *ctxt, const std::string& topModule);

  OpBuilder& builder() { return opBuilder; }
  MLIRContext *context() { return ctxt; }
  void dump() { root.dump(); }
  Value getClock() { return clock; }
  Value getReset() { return reset; }

  void beginContext(Value clock, Value reset, OpBuilder bodyBuilder);
  void endContext();

  void beginModuleDeclaration();
  void endModuleDeclaration();
};

FirpContext *firpContext();
void initFirpContext(MLIRContext *mlirCtxt, const std::string& topModule);

// conventient type constructors

IntType uintType();
IntType uintType(uint32_t bitWidth);
IntType bitType();
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
};

FValue lift(Value val);
FValue cons(uint64_t n, IntType type = IntType::get(firpContext()->context(), false));
FValue mux(FValue cond, FValue pos, FValue neg);
FValue mux(FValue sel, std::initializer_list<FValue> options);
FValue zeros(FIRRTLBaseType type);
FValue ones(FIRRTLBaseType type);

class Reg {
  RegResetOp regOp;
  FIRRTLBaseType type;
public:
  Reg(FIRRTLBaseType type, FValue resetValue, const std::string& name = "");
  Reg(FIRRTLBaseType type, const std::string& name = "");
  FValue read() { return regOp.getResult(); }
  void write(FValue what);
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
  // ensures that there exists exactly one FModuleOp per module class
  static FModuleOp modOp;

  std::string name;
  std::vector<Port> ports;
  std::unordered_map<std::string, uint32_t> portIndices;
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

    modOp = firpContext()->builder().create<FModuleOp>(
      firpContext()->builder().getUnknownLoc(),
      firpContext()->builder().getStringAttr(name),
      portInfos
    );

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
      firpContext()->builder().getStringAttr(name + "Instance")
    );

    // instantiating includes connecting clk and rst
    Value clk = firpContext()->getClock();
    Value rst = firpContext()->getReset();

    // This check is necessary because a module is ALWAYS instantiated, even
    // the top one. The instantiation is only removed afterwards.
    if (clk)
      io("clk") <<= clk;

    if (rst)
      io("rst") <<= rst;
  }
public:
  template <class...Args>
  Module(const std::string& name, std::initializer_list<Port> ports, Args&&...args):
    name(name),
    ports(ports) {

    // insert clock and reset port
    this->ports.insert(
      this->ports.begin(),
      Port("rst", true, bitType()) // ResetType::get(firpContext()->context()))
    );

    this->ports.insert(
      this->ports.begin(),
      Port("clk", true, ClockType::get(firpContext()->context()))
    );

    for (uint32_t i = 0; i < this->ports.size(); ++i)
      portIndices[this->ports[i].name] = i;

    // declare only once
    if (!modOp)
      declare(args...);

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
  }
};

template <class ConcreteModule>
FModuleOp Module<ConcreteModule>::modOp;

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

}