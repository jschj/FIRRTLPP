#pragma once

#include "firp.hpp"


namespace firp {

class TimingAnalysis {
public:
  static uint32_t getEndTimeOf(Value value);

  template <class ConcreteModule>
  static uint32_t getEndTimeOfPort(Module<ConcreteModule> mod, StringRef portName) {
    FModuleOp modOp = mod.getModuleOp();
    size_t portIndex = mod.getPortIndex(portName);
    return getEndTimeOf(modOp.getArgument(portIndex));
  }
};

}