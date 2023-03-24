#pragma once

#include "firp.hpp"


namespace firp {

class FirpQueue : public Module<FirpQueue> {
public:
  FirpQueue(FIRRTLBaseType elementType, size_t depth):
    Module<FirpQueue>(
      "FirpQueue",
      {
        Port("enq", readyValidType(elementType), true),
        Port("deq", readyValidType(elementType), false)
      },
      true,
      elementType, depth
    ) {}
  
  void body(FIRRTLBaseType elementType, size_t depth) {
    size_t indexBits = clog2(depth);

    auto increment = [&](auto value){
      return mux(value == cons(depth - 1), cons(0), value + cons(1));
    };

    auto ram = Memory(elementType, depth);

    auto enqIndex = Reg(uintType(indexBits));
    auto deqIndex = Reg(uintType(indexBits));

    auto nextEnqIndex = increment(enqIndex.read())(indexBits - 1, 0);
    auto nextDeqIndex = increment(deqIndex.read())(indexBits - 1, 0);

    auto wouldEnqOvertake = nextEnqIndex == deqIndex.read();
    auto wouldDeqOvertake = deqIndex.read() == enqIndex.read();

    auto count = Reg(uintType(indexBits + 1));
    auto isEmpty = count.read() == cons(0);

    io("enq")("ready") <<= ~wouldEnqOvertake | isEmpty;
    io("deq")("valid") <<= ~wouldDeqOvertake & ~isEmpty;

    auto enqFire = io("enq")("ready") & io("enq")("valid");
    auto deqFire = io("deq")("ready") & io("deq")("valid");

    when (enqFire, [&](){
      enqIndex.write(nextEnqIndex);
    });

    when (deqFire, [&](){
      deqIndex.write(nextDeqIndex);
    });

    ram.writePort()("addr") <<= enqIndex.read();
    ram.writePort()("en") <<= enqFire;
    ram.writePort()("clk") <<= firpContext()->getClock();
    ram.writePort()("data") <<= io("enq")("bits");
    ram.writePort()("mask") <<= cons(1, bitType());

    ram.readPort()("addr") <<= deqIndex.read();
    ram.readPort()("en") <<= deqFire;
    ram.readPort()("clk") <<= firpContext()->getClock();
    io("deq")("bits") <<= ram.readPort()("data");
  }
};

}