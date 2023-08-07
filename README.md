# FIRRTL++ (FIRRTLPP)
FIRRTL++ is a small C++ library that is supposed to ease the creation of medium-complex hardware designs in CIRCT using the FIRRTL dialect.
It works by mostly abstracting away the repetitive task of calling `opBuilder.create<SomeOp>(...)` and provides a convenient
context-based way of dealing with modules and clock and reset signals. It is strongly inspired by Chisel.

## Disclaimer

This library is in no way complete or well tested. It was developed as part of the Master's thesis.

## Shift Register Example

```
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

void ShiftRegister::body() {
  std::vector<Reg> regs;

  for (size_t i = 0; i < depth; ++i) {
    regs.push_back(elementType);

    if (i >= 1)
      regs[i].write(regs[i - 1].read());
  }

  if (depth >= 1) {
    regs.front().write(io("in"));
    io("out") <<= regs.back().read();
  } else {
    io("out") <<= io("in");
  }
}
```

## Pipelined Integer Adder Example

```
class PipelinedAdder : public firp::Module<PipelinedAdder> {
  uint32_t bitWidth, maxAdderWidth;
public:
  PipelinedAdder(uint32_t bitWidth, uint32_t maxAdderWidth):
    firp::Module<PipelinedAdder>(
      "PipelinedAdder",
      {
        firp::Port("a", true, firp::uintType(bitWidth)),
        firp::Port("b", true, firp::uintType(bitWidth)),
        firp::Port("c", false, firp::uintType(bitWidth + 1))
      },
      bitWidth, maxAdderWidth
    ), bitWidth(bitWidth), maxAdderWidth(maxAdderWidth) { build(); }
  
  void body();

  static uint32_t getDelay(uint32_t bitWidth, uint32_t maxAdderWidth) {
    return bitWidth / maxAdderWidth + (bitWidth % maxAdderWidth ? 1 : 0);
  }

  uint32_t getDelay() const {
    return getDelay(bitWidth, maxAdderWidth);
  }
};
```

```
void PipelinedAdder::body() {
  // inspired by http://vlsigyan.com/pipeline-adder-verilog-code/

  if (bitWidth <= maxAdderWidth) {
    io("c") <<= regNext(io("a") + io("b"));
    return;
  }

  uint32_t stageCount = bitWidth / maxAdderWidth + (bitWidth % maxAdderWidth ? 1 : 0);
  auto carry = cons(0, uintType(1));

  std::vector<FValue> resultChunks;

  for (uint32_t i = 0; i < stageCount; ++i) {
    uint32_t lo = i * maxAdderWidth;
    uint32_t hi = std::min(lo + maxAdderWidth - 1, bitWidth - 1);

    uint32_t preDelay = i;
    uint32_t postDelay = stageCount - i - 1;

    auto inA = wireInit(shiftRegister(io("a")(hi, lo), preDelay), std::string("inA_") + std::to_string(i));
    auto inB = wireInit(shiftRegister(io("b")(hi, lo), preDelay), std::string("inB_") + std::to_string(i));

    // top bit is ignored as it will never be set to 1
    auto sum = regNext(inA.read() + inB + carry).read();
    uint32_t sumWidth = sum.bitCount();

    // top bit will always be 0
    auto sumOut = sum(sumWidth - 3, 0);
    carry = sum(sumWidth - 2);

    resultChunks.push_back(shiftRegister(sumOut, postDelay));
  }

  resultChunks.push_back(carry);
  std::reverse(resultChunks.begin(), resultChunks.end());
  
  io("c") <<= cat(resultChunks);

  // used for debugging
  //svCocoTBVerbatim("PipelinedAdder");
}
```