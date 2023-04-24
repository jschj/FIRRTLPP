#pragma once

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "firp.hpp"


namespace firp::esi {

// Bundle types that contain flipped fields cannot be lowered to HW structs.
// 

// The way the lowering passes for FIRRTL -> HW are designed forces that all 

// This is useful when you want to connect a ESI channel to a module that you
// previously described with FIRRTLPP. Returns a HW type.
Type lowerFIRRTLType(FIRRTLBaseType type);

template <class ConcreteModule>
class ExternalHWModule {
  llvm::hash_code hashValue;
  std::string name;
  std::vector<circt::hw::PortInfo> ports;
  std::unordered_map<std::string, uint32_t> portIndices;
  circt::hw::HWModuleExternOp modOp;
  circt::hw::InstanceOp instOp;

  void declare() {
    OpBuilder& builder = firpContext()->builder();

    // Cannot declare external HW modules inside the firrtl circuit op.
    auto oldPoint = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(
      &firpContext()->root.getBodyRegion().front()
    );

    modOp = firpContext()->builder().create<circt::hw::HWModuleExternOp>(
      firpContext()->builder().getUnknownLoc(),
      firpContext()->builder().getStringAttr(name),
      ports
    );

    builder.restoreInsertionPoint(oldPoint);
  }

  template <class Container>
  void initPorts(Container ports) {
    OpBuilder& builder = firpContext()->builder();

    // insert clock and reset ports
    this->ports.push_back(circt::hw::PortInfo{
      .name = builder.getStringAttr("clock"),
      .direction = circt::hw::PortDirection::INPUT,
      .type = builder.getI1Type()
    });

    this->ports.push_back(circt::hw::PortInfo{
      .name = builder.getStringAttr("reset"),
      .direction = circt::hw::PortDirection::INPUT,
      .type = builder.getI1Type()
    });

    for (const auto& [name, isInput, type] : ports)
      this->ports.push_back(circt::hw::PortInfo{
        .name = builder.getStringAttr(name),
        .direction = isInput ? circt::hw::PortDirection::INPUT : circt::hw::PortDirection::OUTPUT,
        .type = type
      });
  }
public:
  // All Args must be hashable.
  template <class Container>
  ExternalHWModule(const std::string& name, Container ports):
    hashValue(llvm::hash_value(name)),
    name(name) {

    initPorts(ports);

    for (uint32_t i = 0; i < this->ports.size(); ++i)
      portIndices[this->ports[i].name.str()] = i;
    
    declare();
  }

  // HW instances work a little different
  void instantiateWithInputs(const std::vector<Value>& values) {
    instOp = firpContext()->builder().create<circt::hw::InstanceOp>(
      firpContext()->builder().getUnknownLoc(),
      modOp,
      firpContext()->builder().getStringAttr(name + "_instance"),
      ArrayRef<Value>(values)
    );
  }
};

// Lowering FIRRTL to HW poses the following problem: Aggregate types with
// flipped fields can only be lowered to unidirectional types. Therefore,
// all types are broken down to ground types (--lower-firrtl-types).
// We need a way to make a bundle type primitive so that it survives --lower-firrtl-types.

std::vector<Port> toPrimitivePorts(const std::string& stemName, bool isInput, FIRRTLBaseType type);

// We break down composite types into their primitive parts when defining the io
// ports of modules. We do this because primitive types survive the lowering pass
// to HW and we need this predictable behaviour to make them work with ESI.
// On the other hand we cannot bundle them back together and provide sub field
// accesss as this is not supported for writing. This class should provide a
// way of performing "virtual" sub field access akin to the usual .io("...")
// known from normal modules.
class ConnectionHelperObject {
  FIRRTLBaseType type;
  
public:
  ConnectionHelperObject operator()(const std::string& fieldName);
  operator FValue();
  void operator<<=(FValue what);
};

template <class ConcreteModule>
class ESIModule : public ExternalModule<ConcreteModule> {
protected:
  FIRRTLBaseType highType;
  std::string uniqueName;

  void connect(FValue wire, bool isInput, const std::string& stem) {
    if (IntType intType = wire.getType().dyn_cast<IntType>()) {
      // If wire is a primitive type we are done. Then we connect the stem
      // (such as "deq_valid") to the wire directly.
      if (isInput)
        ExternalModule<ConcreteModule>::io(stem) <<= wire;
      else
        wire <<= ExternalModule<ConcreteModule>::io(stem);

    } else if (BundleType bundleType = wire.getType().dyn_cast<BundleType>()) {
      // If wire is a bundle type we need to iterate over its components.
      for (const BundleType::BundleElement& el : bundleType) {
        std::string newStem = stem + "_" + el.name.str();
        connect(wire(el.name.str()), isInput != el.isFlip, newStem);
      }
    } else {
      assert(false);
    }
  }

  static Type uintType(uint32_t bitWidth);

public:
  ESIModule(const std::string& baseName, FIRRTLBaseType highType, const std::string& portBaseName, bool isInput):
    ExternalModule<ConcreteModule>(
      baseName + "_" + std::to_string(mlir::hash_value(highType)),
      toPrimitivePorts(portBaseName, isInput, highType)
    ),
    highType(highType),
    uniqueName(baseName + "_" + std::to_string(mlir::hash_value(highType))) {}

  std::string getUniqueName() const {
    return uniqueName;
  }
};

// hw.module.extern @ESIReceiver_390360(%clock: i1, %reset: i1, %deq_ready: i1) -> (deq_valid: i1, deq_bits_bits: i40, deq_bits_last: i1)
// hw.module.extern @ESISender_390360(%clock: i1, %reset: i1, %enq_valid: i1, %enq_bits_bits: i40, %enq_bits_last: i1) -> (enq_ready: i1)

class ESIReceiver : public ESIModule<ESIReceiver> {
  FIRRTLBaseType innerType;
public:
  ESIReceiver(FIRRTLBaseType innerType):
    ESIModule<ESIReceiver>(
      "ESIReceiver",
      readyValidType(innerType),
      "deq",
      false
    ),
    innerType(innerType) {}

  // We provide a custom implementation of io() because we have broken up the nice
  // bundle into its primitive components but still want to retain a high level
  // interface.
  FValue io(const std::string& name);
};

class ESISender : public ESIModule<ESISender> {
public:
  ESISender(FIRRTLBaseType innerType):
    ESIModule<ESISender>(
      "ESISender",
      readyValidType(innerType),
      "enq",
      true
    ) {}

  // We provide a custom implementation of io() because we have broken up the nice
  // bundle into its primitive components but still want to retain a high level
  // interface.
  FValue io(const std::string& name);
};

}