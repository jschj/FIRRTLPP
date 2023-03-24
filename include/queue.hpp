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
    assert(depth >= 1);
    uint32_t countBits = clog2(depth);
    uint32_t depthBits = clog2(depth - 1);

    auto ringBuf = Vector(elementType, depth);

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

    auto doesEnq = io("enq")("valid") & io("enq")("ready");
    auto doesDeq = io("deq")("valid") & io("deq")("ready");

    when (doesEnq, [&](){
      enqIndex << Const(1) + enqIndex;
      ringBuf.write(deqIndex, io("enq")("bits"));
    }).build();

    when (doesDeq, [&](){
      deqIndex << Const(1) + deqIndex;
    }).build();

    io("deq")("bits") << ringBuf(deqIndex);

    auto count = Reg(UInt(countBits));
    count << count.val() +
      Mux(doesEnq, Const(1), Const(0)) -
      Mux(doesDeq, Const(1), Const(0));

    io("count") << count;
  }
};

}