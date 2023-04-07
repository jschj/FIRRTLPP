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
// previously described with FIRRTLPP.
Type lowerFIRRTLType(FIRRTLBaseType type);

void unwrapAndConnect(circt::hw::HWModuleOp dst, circt::esi::ChannelType srcChan);

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

}