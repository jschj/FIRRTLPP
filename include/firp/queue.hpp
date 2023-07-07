#pragma once

#include "firp.hpp"


namespace firp {

class Queue : public Module<Queue> {
  FIRRTLBaseType innerType;
  uint32_t size;

  FValue buildRAM(FValue writeAddr, FValue writeData, FValue writeEn,
                  FValue readAddr, FValue readEn);
public:
  Queue(FIRRTLBaseType innerType, uint32_t size):
    Module<Queue>(
      "Queue",
      {
        Input("enq", readyValidType(innerType)),
        Output("deq", readyValidType(innerType))
      },
      innerType, size
    ),
    innerType(innerType), size(size) { build(); }

  void body();
};

}