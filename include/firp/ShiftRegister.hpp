#pragma once

#include "firp.hpp"


namespace firp {

class ShiftRegister : public Module<ShiftRegister> {
  FIRRTLBaseType elementType;
  size_t depth;
public:
  ShiftRegister(FIRRTLBaseType elementType, size_t depth):
    Module<ShiftRegister>(
      "ShiftRegister",
      {
        Input("in", elementType),
        Output("out", elementType)
      },
      elementType, depth
    ),
    elementType(elementType), depth(depth) { build(); }

  void body();
};

}