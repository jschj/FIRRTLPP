#pragma once

#include "firp.hpp"


namespace ufloat {

struct UFloatConfig {
  uint32_t exponentWidth;
  uint32_t mantissaWidth;

  uint32_t getWidth() const { return exponentWidth + mantissaWidth; }
  firp::FIRRTLBaseType getType() const;
};

firp::FIRRTLBaseType ufloatType(const UFloatConfig& cfg);

struct UFloat : public firp::FValue {
  // TODO: Maybe throw this away?
};

}

namespace llvm {

template <>
struct DenseMapInfo<ufloat::UFloatConfig> {

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

}

namespace ufloat {

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

class FPAdd : public firp::Module<FPAdd> {
  UFloatConfig cfg;
public:
  FPAdd(const UFloatConfig& cfg):
    firp::Module<FPAdd>(
      "FPAdd",
      {
        firp::Port("a", true, firp::uintType(cfg.getWidth())),
        firp::Port("b", true, firp::uintType(cfg.getWidth())),
        firp::Port("c", false, firp::uintType(cfg.getWidth()))
      },
      cfg
    ), cfg(cfg) { build(); }

  void body();
};

}