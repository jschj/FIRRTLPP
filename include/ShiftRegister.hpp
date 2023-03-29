#pragma once

#include "firp.hpp"


namespace firp {

class ShiftRegister : public Module<ShiftRegister> {
public:
  ShiftRegister(FIRRTLBaseType elementType, size_t depth):
    Module<ShiftRegister>(
      "ShiftRegister",
      {
        Port("in", true, elementType),
        Port("out", false, elementType)
      },
      elementType, depth
    ) {}

  void body(FIRRTLBaseType elementType, size_t depth);
};

/*

  class ShiftRegister : public Module<ShiftRegister> {
  public:

    ShiftRegister(FIRRTLBaseType elementType, size_t depth):
      Module<ShiftRegister>(elementType, depth) {}

    std::string name() const { return "ShiftRegister"; }

    std::initializer_list<Port> ports() const {
      return {
        ...
      };
    }

  };

  auto shiftReg = Module<ShiftRegister>(elType, depth);


 */

}