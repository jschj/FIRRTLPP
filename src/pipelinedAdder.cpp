#include <firp/pipelinedAdder.hpp>


namespace ufloat {

using namespace firp;

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
    llvm::outs() << " sum/inA/inB width: " << sum.bitCount() << " " << inA.read().bitCount() << " " << inB.read().bitCount() << "\n";

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

}