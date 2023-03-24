#pragma once

#include <memory>

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

namespace firrtlpp {

using namespace ::mlir;
using namespace ::circt::firrtl;

class PrimitiveBuilder {
public:
  MLIRContext *context;
  OpBuilder builder;

  ModuleOp root;
  CircuitOp circuitOp;
  Block *lastInsertionBlock;

  Value clk;
public:
  PrimitiveBuilder(MLIRContext *context, const std::string& topModule):
    context(context),
    builder(context) {

    root = builder.create<ModuleOp>(
      builder.getUnknownLoc()
    );

    builder.setInsertionPointToStart(
      &root.getBodyRegion().front()
    );

    circuitOp = builder.create<CircuitOp>(
      builder.getUnknownLoc(),
      builder.getStringAttr(topModule)
    );

    builder = circuitOp.getBodyBuilder();
  }

  void setClock(Value value) {
    clk = value;
  }

  Value getClock() {
    return clk;
  }

  void dump() {
    assert(succeeded(root.verify()));
    root.dump();
  }

  void beginModule() {
    lastInsertionBlock = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
  }

  void endModule() {
    builder.setInsertionPointToEnd(lastInsertionBlock);
  }
};

inline std::unique_ptr<PrimitiveBuilder> prim;

PrimitiveBuilder *getPrimitiveBuilder();
void initPrimitiveBuilder(MLIRContext *ctxt, const std::string& topModule);

inline StringAttr strAttr(const std::string& s) {
  return prim->builder.getStringAttr(s);
}

Value constant(int64_t value, uint32_t bitWidth);

inline IntType Bit() {
  return IntType::get(prim->context, false, 1);
}

inline IntType UInt() {
  return IntType::get(prim->context, false);
}

inline IntType UInt(uint32_t bitWidth) {
  return IntType::get(prim->context, false, bitWidth);
}

//inline Value UInt(uint32_t bitWidth, uint64_t value) {
//  return prim->builder.create<ConstantOp>(
//    prim->builder.getUnknownLoc(),
//    UInt(bitWidth),
//    ::llvm::APInt(bitWidth, value)
//  ).getResult();
//}

inline IntType SInt() {
  return IntType::get(prim->context, true);
}

inline IntType SInt(uint32_t bitWidth) {
  return IntType::get(prim->context, true, bitWidth);
}

inline BundleType Bundle(ArrayRef<BundleType::BundleElement> elements) {
  return BundleType::get(elements, prim->context);
}



// can be used to build any kind of expression

/*
  Value a, b, c, d;
  ...
  Value e = (  (a & b) | c + d  ).build(primBuilder);

  auto x = ...;

  // Problem: x can be any expression-ptr and access operators cannot be defined
  // outside the class they operate on.
  auto isValid = x["bits"]["valid"];


 */

class Expression {
public:
  enum Operation {
    OP_AND, OP_OR, OP_NEG, OP_ADD, OP_SUB, OP_GT, OP_GEQ,
    OP_LT, OP_LEQ, OP_EQ, OP_NEQ
  };

  virtual ~Expression() {}
  virtual Value build() const = 0;
};

// This class ensures that everything lands on the heap and that the references
// therefore not point to destructed stack objects.
class ExpressionWrapper {
  std::shared_ptr<Expression> ptr;
public:
  ExpressionWrapper() {}
  ExpressionWrapper(std::shared_ptr<Expression> ptr): ptr(ptr) {}

  template <class T, class ...Args>
  static ExpressionWrapper make(Args&&...args) {
    return ExpressionWrapper(std::make_shared<T>(args...));
  }

  Value build() const {
    return ptr->build();
  }

  // We can now implement all the operators we couldn't before.

  ExpressionWrapper operator~() const;

  ExpressionWrapper operator&(ExpressionWrapper b) const;
  ExpressionWrapper operator|(ExpressionWrapper b) const;
  ExpressionWrapper operator+(ExpressionWrapper b) const;
  ExpressionWrapper operator-(ExpressionWrapper b) const;
  ExpressionWrapper operator>(ExpressionWrapper b) const;
  ExpressionWrapper operator>=(ExpressionWrapper b) const;
  ExpressionWrapper operator<(ExpressionWrapper b) const;
  ExpressionWrapper operator<=(ExpressionWrapper b) const;
  ExpressionWrapper operator==(ExpressionWrapper b) const;
  ExpressionWrapper operator!=(ExpressionWrapper b) const;

  // field
  ExpressionWrapper operator()(const std::string& fieldName) const;
  // bits
  ExpressionWrapper operator()(size_t hi, size_t lo) const;
  // single bit
  ExpressionWrapper operator()(size_t bitIndex) const;
};

class ValueExpression : public Expression {
  Value value;
public:
  // lift value to expression
  ValueExpression(Value value): value(value) {}

  Value build() const override {
    return value;
  }
};

inline ExpressionWrapper lift(Value val) {
  return ExpressionWrapper::make<ValueExpression>(val);
}

template <class T>
inline ExpressionWrapper Const(T value, IntType type = IntType::get(prim->context, false));

template <>
inline ExpressionWrapper Const(int value, IntType type) {
  ::llvm::APInt concreteValue =
    type.getWidthOrSentinel() == -1 ?
    ::llvm::APInt(64, value) :
    ::llvm::APInt(type.getWidthOrSentinel(), value);

  Value val = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    type,
    concreteValue
  ).getResult();

  return ExpressionWrapper::make<ValueExpression>(val);
}

template <>
inline ExpressionWrapper Const(uint32_t value, IntType type) {
  ::llvm::APInt concreteValue =
    type.getWidthOrSentinel() == -1 ?
    ::llvm::APInt(64, value) :
    ::llvm::APInt(type.getWidthOrSentinel(), value);

  Value val = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    type,
    concreteValue
  ).getResult();

  return ExpressionWrapper::make<ValueExpression>(val);
}

inline ExpressionWrapper UInt(int64_t value, uint32_t bitWidth) {
  Value val = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    UInt(bitWidth),
    ::llvm::APInt(bitWidth, value)
  ).getResult();

  return ExpressionWrapper::make<ValueExpression>(val);
}

inline ExpressionWrapper SInt(int64_t value, uint32_t bitWidth) {
  Value val = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    SInt(bitWidth),
    ::llvm::APInt(bitWidth, value)
  ).getResult();

  return ExpressionWrapper::make<ValueExpression>(val);
}

class UnaryExpression : public Expression {
  ExpressionWrapper operand;
  Operation operation;
public:
  UnaryExpression(UnaryExpression&) = delete;
  UnaryExpression(const UnaryExpression&) = delete;
  UnaryExpression(UnaryExpression&&) = delete;

  UnaryExpression(ExpressionWrapper operand, Operation operation):
    operand(operand),
    operation(operation) {}

  Value build() const override {
    Value input = operand.build();
    Value output;

    switch (operation) {
      case OP_NEG:
        output = prim->builder.create<NotPrimOp>(
          prim->builder.getUnknownLoc(),
          input
        ).getResult();
        break;
      default:
        assert(false && "invalid unary operation");
    }

    assert(output);
    return output;
  }
};

class BinaryExpression : public Expression {
  ExpressionWrapper lhs;
  ExpressionWrapper rhs;
  Operation operation;
public:
  //BinaryExpression(BinaryExpression&) = delete;
  //BinaryExpression(const BinaryExpression&) = delete;
  //BinaryExpression(BinaryExpression&&) = delete;

  BinaryExpression(ExpressionWrapper lhs, ExpressionWrapper rhs, Operation operation):
    lhs(lhs),
    rhs(rhs),
    operation(operation) {}

  Value build() const override {
    Value leftInput = lhs.build();
    Value rightInput = rhs.build();
    Location loc = prim->builder.getUnknownLoc();
    Value output;

    switch (operation) {
      case OP_AND:
        output = prim->builder.create<AndPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_OR:
        output = prim->builder.create<OrPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_ADD:
        output = prim->builder.create<AddPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_SUB:
        output = prim->builder.create<SubPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_GT:
        output = prim->builder.create<GTPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_GEQ:
        output = prim->builder.create<GEQPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_LT:
        output = prim->builder.create<LTPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_LEQ:
        output = prim->builder.create<LEQPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_EQ:
        output = prim->builder.create<EQPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      case OP_NEQ:
        output = prim->builder.create<NEQPrimOp>(loc, leftInput, rightInput).getResult();
        break;
      default:
        assert(false && "invalid binary operation");
    }

    assert(output);
    return output;
  }
};

class BitsExpression : public Expression {
  ExpressionWrapper operand;
  uint32_t hi, lo;
public:
  BitsExpression(ExpressionWrapper operand, uint32_t hi, uint32_t lo):
    operand(operand), hi(hi), lo(lo) {}

  Value build() const override {
    return prim->builder.create<BitsPrimOp>(
      prim->builder.getUnknownLoc(),
      operand.build(),
      prim->builder.getI32IntegerAttr(hi),
      prim->builder.getI32IntegerAttr(lo)
    ).getResult();
  }
};

inline ExpressionWrapper bits(ExpressionWrapper of, uint32_t hi, uint32_t lo) {
  return ExpressionWrapper::make<BitsExpression>(of, hi, lo);
}

class FieldExpression : public Expression {
  ExpressionWrapper operand;
  std::string fieldName;
public:
  FieldExpression(ExpressionWrapper operand, const std::string& fieldName):
    operand(operand), fieldName(fieldName) {}

  Value build() const override {
    return prim->builder.create<SubfieldOp>(
      prim->builder.getUnknownLoc(),
      operand.build(),
      fieldName
    ).getResult();
  }
};

inline ExpressionWrapper field(ExpressionWrapper of, const std::string& fieldName) {
  return ExpressionWrapper::make<FieldExpression>(of, fieldName);
}

class BundleExpression : public Expression {
  std::vector<ExpressionWrapper> operands;
  BundleType bundleType;
public:
  template <class Container>
  BundleExpression(BundleType bundleType, Container operands):
    operands(std::begin(operands), std::end(operands)),
    bundleType(bundleType) {}

  Value build() const override {
    std::vector<Value> inputs;
    for (const auto& op : operands)
      inputs.push_back(op.build());

    return prim->builder.create<BundleCreateOp>(
      prim->builder.getUnknownLoc(),
      bundleType,
      ValueRange(inputs)
    ).getResult();
  }
};

class MuxExpression : public Expression {
  ExpressionWrapper cond, positive, negative;
public:
  MuxExpression(ExpressionWrapper cond, ExpressionWrapper positive, ExpressionWrapper negative):
    cond(cond), positive(positive), negative(negative) {}

  Value build() const override {
    return prim->builder.create<MuxPrimOp>(
      prim->builder.getUnknownLoc(),
      cond.build(),
      positive.build(),
      negative.build()
    ).getResult();
  }
};

inline ExpressionWrapper Mux(ExpressionWrapper cond, ExpressionWrapper positive, ExpressionWrapper negative) {
  return ExpressionWrapper::make<MuxExpression>(cond, positive, negative);
}

// This allows connecting together arbitrary expressions. However, firrtl has a concept
// of source/sink/duplex flow. So be wary of what you connect together!
inline void operator<<(ExpressionWrapper dst, ExpressionWrapper src) {
  Value input = src.build();
  Value output = dst.build();

  prim->builder.create<StrictConnectOp>(
    prim->builder.getUnknownLoc(),
    output,
    input
  );
}

inline ExpressionWrapper zero(FIRRTLBaseType type) {
  int32_t bitWidth = type.getBitWidthOrSentinel();
  assert(bitWidth != -1 && "type has unknown bit width");
  IntType inputType = IntType::get(prim->context, false, bitWidth);

  ConstantOp zeroConstant = prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    inputType,
    ::llvm::APInt(bitWidth, 0)
  );

  Value result = prim->builder.create<BitCastOp>(
    prim->builder.getUnknownLoc(),
    type,
    zeroConstant
  );

  return lift(result);
}

inline void operator<<(ExpressionWrapper dst, int n) {
  assert(n == 0 && "can only be used for 0 initialization");
  Value output = dst.build();
  Value input = zero(dyn_cast<FIRRTLBaseType>(output.getType())).build();

  prim->builder.create<StrictConnectOp>(
    prim->builder.getUnknownLoc(),
    output,
    input
  );
}

class Statement {
};

class Reg : public Statement {
  RegOp regOp;
public:
  Reg(Type type) {
    regOp = prim->builder.create<RegOp>(
      prim->builder.getUnknownLoc(),
      type,
      prim->getClock()
    );
  }

  //Reg(const Reg& other) = delete;

  operator Value() {
    return regOp.getResult();
  }

  operator ExpressionWrapper() {
    return lift(*this);
  }

  ExpressionWrapper val() {
    return lift(*this);
  }

  void operator<<(ExpressionWrapper what) {
    Value input = what.build();
    prim->builder.create<ConnectOp>(
      prim->builder.getUnknownLoc(),
      regOp,
      input
    );
  }
};

// ::circt::hw::PortInfo doesn't really fit into our design
struct Port {
  enum Direction { Input, Output };

  Direction direction;
  std::string name;
  Type type;

  Port() {}
  Port(Direction direction, const std::string& name, Type type):
    direction(direction), name(name), type(type) {}
};

template <class ConcreteModule>
class Module {
protected:
  // The template is a trick to enforce that modOp exists exactly once per concrete module class.
  static FModuleOp modOp;

  std::string moduleName;
  std::vector<Port> ports;
  bool hasClock;
  std::unordered_map<std::string, size_t> portIndices;
  //std::unordered_map<std::string, size_t> inPortIndices;
  //std::unordered_map<std::string, size_t> outPortIndices;

  InstanceOp instOp;

  template <class...Args>
  void declareOnce(Args&&...args) {
    // check if it has already been declared
    if (modOp)
      return;

    // if not, declare
    size_t portIndex = 0;
    for (const Port& port : ports)
      portIndices[port.name] = portIndex++;

    std::vector<PortInfo> portInfos;
    for (const Port& port : ports)
      portInfos.emplace_back(
        prim->builder.getStringAttr(port.name),
        port.type,
        port.direction == Port::Direction::Input ? Direction::In : Direction::Out
      );

    Block *lastInsertion = prim->builder.getInsertionBlock();
    prim->builder.setInsertionPointToEnd(prim->circuitOp.getBodyBlock());

    modOp = prim->builder.create<FModuleOp>(
      prim->builder.getUnknownLoc(),
      prim->builder.getStringAttr(moduleName),
      portInfos
    );

    Block *bodyBlock = modOp.getBodyBlock();
    prim->builder.setInsertionPointToEnd(bodyBlock);

    if (hasClock)
      prim->setClock(bodyBlock->getArguments()[0]);

    static_cast<ConcreteModule *>(this)->body(args...);

    prim->builder.setInsertionPointToEnd(lastInsertion);
  }

  void instantiate() {
    instOp = prim->builder.create<InstanceOp>(
      prim->builder.getUnknownLoc(),
      modOp,
      prim->builder.getStringAttr("TestModuleInstance")
    );
  }
protected:
  template <class...Args>
  Module(const std::string& moduleName, std::initializer_list<Port> ports, bool hasClock, bool isTop, Args&&...args):
    moduleName(moduleName),
    ports(ports),
    hasClock(hasClock) {
    // This constructor is used for declaration and instantiation at the same time.

    // By default every module has the clk as its first argument.
    if (hasClock) {
      this->ports.insert(
        this->ports.begin(),
        Port(Port::Direction::Input, "clk", ClockType::get(prim->context))
      );
    }
    
    // Declaration can happen at most once.
    declareOnce(args...);

    if (!isTop)
      instantiate();
  }

  Value getArgument(const std::string& name) {
    size_t index = portIndices.at(name);
    return modOp.getBodyBlock()->getArgument(index);
  }

  ExpressionWrapper arg(const std::string& name) {
    return lift(getArgument(name));
  }

  // has a very similar interface to Reg
  class ArgRef {
    Module<ConcreteModule> *mod;
    Value val;
    bool isInput;

    ArgRef(Module<ConcreteModule> *mod, const std::string& name): mod(mod) {
      size_t index = mod->portIndices.at(name);
      val = mod->modOp.getBodyBlock()->getArgument(index);
      isInput = mod->ports[index].direction == Port::Direction::Input;
    }

    operator ExpressionWrapper() {
      //assert(isInput);
      return lift(val);
    }

    void operator<<(ExpressionWrapper what) {
      assert(!isInput);
      Value input = what.build();
      prim->builder.create<ConnectOp>(
        prim->builder.getUnknownLoc(),
        val,
        input
      );
    }
  };

  //ArgRef io(const std::string& name) const {
  //  return ArgRef(this, name);
  //}

  ExpressionWrapper io(const std::string& name) const {
    size_t index = portIndices.at(name);
    return lift(modOp.getBodyBlock()->getArgument(index));
  }
public:
  virtual ~Module() {}
};

template <class T>
FModuleOp Module<T>::modOp;

inline BundleType ReadyValidIO(FIRRTLBaseType elementType) {
  return BundleType::get(
    {
      BundleType::BundleElement(strAttr("ready"), true, Bit()),
      BundleType::BundleElement(strAttr("valid"), false, Bit()),
      BundleType::BundleElement(strAttr("bits"), false, elementType)
    },
    prim->context
  );
}

//class Memory : public Module<Memory> {
//
//
//public:
//  Memory() {
//
//  }
//};

template <class ScalarType>
inline std::enable_if_t<std::is_scalar_v<ScalarType>, ScalarType> clog2(const ScalarType& value) {
  ScalarType n(1);
  ScalarType bits(0);

  while (n < value) {
    ++bits;
    n <<= 1;
  }

  return bits;
}



class Conditional {
public:
  typedef std::function<void()> BodyCtor;
private:
  std::vector<std::tuple<ExpressionWrapper, BodyCtor>> whens;
  Optional<BodyCtor> otherwiseCtor;
public:
  Conditional(ExpressionWrapper condition, BodyCtor bodyCtor) {
    whens.push_back(std::make_tuple(condition, bodyCtor));
  }

  Conditional& elseWhen(ExpressionWrapper condition, BodyCtor bodyCtor) {
    assert(!otherwiseCtor.has_value());
    whens.push_back(std::make_tuple(condition, bodyCtor));
    return *this;
  }

  Conditional otherwise(BodyCtor bodyCtor) {
    assert(!otherwiseCtor.has_value());
    otherwiseCtor = bodyCtor;
    return *this;
  }

  void build(int32_t i, OpBuilder builder) {
    // TODO: otherwise
    bool isLast = i == whens.size() - 1;
    auto [cond, bodyCtor] = whens[i];

    WhenOp whenOp = builder.create<WhenOp>(
      builder.getUnknownLoc(),
      cond.build(),
      !isLast || otherwiseCtor.has_value()
    );
    
    OpBuilder old = prim->builder;
    prim->builder = whenOp.getThenBodyBuilder();
    bodyCtor();
    prim->builder = old;

    if (!isLast) {      
      OpBuilder old = prim->builder;
      prim->builder = whenOp.getThenBodyBuilder();
      bodyCtor();
      prim->builder = old;

      build(i + 1, whenOp.getElseBodyBuilder());
    } else if (otherwiseCtor.has_value()) {
      OpBuilder old = prim->builder;
      prim->builder = whenOp.getElseBodyBuilder();
      otherwiseCtor.value()();
      prim->builder = old;
    }
  }

  void build() {
    build(0, prim->builder);
  }
};

inline Conditional when(ExpressionWrapper condition, Conditional::BodyCtor bodyCtor) {
  return Conditional(condition, bodyCtor);
}

class Vector {
  std::vector<Reg> regs;
public:
  Vector(FIRRTLBaseType elementType, size_t count) {
    for (size_t i = 0; i < count; ++i)
      regs.push_back(Reg(elementType));
  }

  void write(ExpressionWrapper index, ExpressionWrapper what) {
    for (int i = 0; i < regs.size(); ++i) {
      when (Const(i) == index, [&](){
        regs[i] << what;
      }).otherwise([&](){
        regs[i] << regs[i];
      }).build();
    }
  }

  ExpressionWrapper operator()(ExpressionWrapper index) {
    std::vector<Value> possibleValues;
    for (size_t i = 0; i < regs.size(); ++i)
      possibleValues.push_back(regs[i].val().build());

    MultibitMuxOp op = prim->builder.create<MultibitMuxOp>(
      prim->builder.getUnknownLoc(),
      index.build(),
      ArrayRef(possibleValues)
    );

    return lift(op);
  }
};

}