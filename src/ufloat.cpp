#include "ufloat.hpp"


template <>
struct llvm::DenseMapInfo<ufloat::UFloatConfig> {
  typedef std::tuple<uint32_t, uint32_t> tuple_t;
private:
  static tuple_t toTuple(const ufloat::UFloatConfig& val) {
    return std::make_tuple(val.exponentWidth, val.mantissaWidth);
  }

  static ufloat::UFloatConfig fromTuple(const tuple_t& val) {
    return ufloat::UFloatConfig {
      .exponentWidth = std::get<0>(val),
      .mantissaWidth = std::get<1>(val)
    };
  }
public:
  static inline ufloat::UFloatConfig getEmptyKey() { return fromTuple(DenseMapInfo<tuple_t>::getEmptyKey()); }
  static inline ufloat::UFloatConfig getTombstoneKey() { return fromTuple(DenseMapInfo<tuple_t>::getTombstoneKey()); }

  static unsigned getHashValue(const ufloat::UFloatConfig& Val) {
    return DenseMapInfo<tuple_t>::getHashValue(toTuple(Val));
  }

  static bool isEqual(const ufloat::UFloatConfig& LHS,
                      const ufloat::UFloatConfig& RHS) {
    return DenseMapInfo<tuple_t>::isEqual(toTuple(LHS), toTuple(RHS));
  }
};

namespace ufloat {

using namespace firp;


void PipelinedAdder::body() {
  // inspired by http://vlsigyan.com/pipeline-adder-verilog-code/

  uint32_t stageCount = bitWidth / maxAdderWidth + (bitWidth % maxAdderWidth ? 1 : 0);
  auto carry = cons(0, uintType(1));
  //Wire result(uintType(bitWidth + 1));
  std::vector<FValue> resultChunks;

  for (uint32_t i = 0; i < stageCount; ++i) {
    uint32_t lo = i * maxAdderWidth;
    uint32_t hi = std::min(lo + maxAdderWidth - 1, bitWidth - 1);

    uint32_t preDelay = i;
    uint32_t postDelay = stageCount - i - 1;

    //llvm::outs() << "i=" << i << " preDelay + postDelay = " << preDelay << " " << postDelay << "\n";
    //llvm::outs() << "hi, lo " << hi << " " << lo << "\n";

    /*
    llvm::outs() << std::string("inA_") + std::to_string(i) << " is delayed by " << preDelay << "\n";

    auto inA = wireInit(shiftRegister(io("a")(hi, lo), preDelay), std::string("inA_") + std::to_string(i));
    auto inB = wireInit(shiftRegister(io("b")(hi, lo), preDelay), std::string("inB_") + std::to_string(i));

    // top bit is ignored as it will never be set to 1
    auto sum = inA.read() + inB + shiftRegister(carry, 1);

    auto sumOut = sum(maxAdderWidth - 1, 0);
    carry = sum(maxAdderWidth);

    resultChunks.push_back(shiftRegister(sumOut, postDelay));
     */

    auto aDelayed = wireInit(shiftRegister(io("a")(hi, lo), preDelay + postDelay), std::string("aDelayed_") + std::to_string(i));
    resultChunks.push_back(aDelayed);
  }

  resultChunks.push_back(carry);
  std::reverse(resultChunks.begin(), resultChunks.end());

  
  io("c") <<= cat(resultChunks);
}

}