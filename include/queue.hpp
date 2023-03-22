#pragma once

#include "firrtlpp.hpp"


namespace firrtlpp {

class SmallQueue : public Module<SmallQueue> {
public:
  SmallQueue(FIRRTLBaseType elementType, uint32_t depth):
    Module<SmallQueue>(
      "SmallQueue",
      {
        Port(Port::Direction::Input, "enq", ReadyValidIO(elementType)),
        Port(Port::Direction::Output, "deq", ReadyValidIO(elementType)),
        Port(Port::Direction::Output, "count", UInt(clog2(depth)))
      }, true, true, elementType, depth)
       {

  }

  void body(FIRRTLBaseType elementType, uint32_t depth) {
    // TODO: bit width inference!
    uint32_t depthBits = clog2(depth - 1);

    std::vector<Reg> ringBuf;
    for (uint32_t i = 0; i < depth; ++i)
      ringBuf.emplace_back(elementType);

    auto maxIndex = Const(depth - 1);

    auto inc = [&](auto n) {
      auto next = Const(1) + n;
      return Mux(maxIndex == next, Const(0), next);
    };

    Reg enqIndex(UInt(depthBits));
    auto nextEnqIndex = inc(enqIndex);

    Reg deqIndex(UInt(depthBits));
    auto nextDeqIndex = inc(deqIndex);

    auto wouldEnqOvertake = nextEnqIndex == deqIndex;
    auto wouldDeqOvertake = deqIndex.val() == enqIndex;    

    io("enq")("ready") << ~wouldEnqOvertake;
    io("deq")("valid") << ~wouldDeqOvertake;

    when (io("enq")("valid") & io("enq")("ready"), [&](){
      // TODO: actually insert element
      enqIndex << Const(1) + enqIndex;
    }).build();

    when (io("deq")("valid") & io("deq")("ready"), [&](){
      // TODO: actually remove element
      deqIndex << Const(1) + deqIndex;
    }).build();
  }
};

}