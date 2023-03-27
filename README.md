# FIRRTL++ (FIRRTLPP)
FIRRTL++ is a small C++ library that is supposed to ease the creation of medium-complex hardware designs in CIRCT using the FIRRTL dialect.
It works by mostly abstracting away the repetitive task of calling `opBuilder.create<SomeOp>(...)` and provides a convenient
context-based way of dealing with modules and clock and reset signals.

This library is in no way complete or tested! New features are added only if they are required.

# Example
A implementation of a simple queue can look like this. As you can see it is stronly insipred by Chisel.

```
class FirpQueue : public Module<FirpQueue> {
public:
  FirpQueue(FIRRTLBaseType elementType, size_t depth):
    Module<FirpQueue>(
      "FirpQueue",
      {
        Port("enq", true, readyValidType(elementType)),
        Port("deq", false, readyValidType(elementType))
      },
      elementType, depth
    ) {}
  
  void body(FIRRTLBaseType elementType, size_t depth) {
    // This hack is easier than implementing simultaneous enqueue and dequeue logic.
    depth += 1;

    size_t indexBits = clog2(depth - 1);

    auto increment = [&](auto value){
      return mux(value == cons(depth - 1), cons(0), value + cons(1));
    };

    auto ram = Memory(elementType, depth);

    auto enqIndex = Reg(uintType(indexBits), "enqIndex");
    auto deqIndex = Reg(uintType(indexBits), "deqIndex");

    auto nextEnqIndex = increment(enqIndex.read())(indexBits - 1, 0);
    auto nextDeqIndex = increment(deqIndex.read())(indexBits - 1, 0);

    auto wouldEnqOvertake = nextEnqIndex == deqIndex.read();
    auto wouldDeqOvertake = deqIndex.read() == enqIndex.read();

    auto count = Reg(uintType(indexBits + 1), "count");
    auto isEmpty = count.read() == cons(0);

    io("enq")("ready") <<= ~wouldEnqOvertake | isEmpty;
    io("deq")("valid") <<= ~wouldDeqOvertake & ~isEmpty;

    auto enqFire = io("enq")("ready") & io("enq")("valid");
    auto deqFire = io("deq")("ready") & io("deq")("valid");

    count.write(
      count.read()
      + mux(enqFire, cons(1), cons(0))
      - mux(deqFire, cons(1), cons(0))
    );

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
    ram.writePort()("mask") <<= ram.maskEnable();

    ram.readPort()("addr") <<= deqIndex.read();
    ram.readPort()("en") <<= deqFire;
    ram.readPort()("clk") <<= firpContext()->getClock();
    io("deq")("bits") <<= ram.readPort()("data");

    svVerbatim(R"(
`ifdef COCOTB_SIM
  initial begin
    $dumpfile("FirpQueue.vcd");
    $dumpvars (0, FirpQueue);
    #1;
  end
`endif
)");
  }
};
```
