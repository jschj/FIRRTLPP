#pragma once

#include "pipelinedAdder.hpp"
#include "dspMult.hpp"


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

struct DSPMultAllocation {
  uint32_t xPos;
  uint32_t yPos;
  uint32_t xWidth;
  uint32_t yWidth;

  uint32_t shift() const { return xPos + yPos; }
  uint32_t xSelectorA() const { return xPos; }
  uint32_t xSelectorB() const { return xPos + xWidth - 1; }
  uint32_t ySelectorA() const { return yPos; }
  uint32_t ySelectorB() const { return yPos + yWidth - 1; }
};

class FPMult : public firp::Module<FPMult> {
  UFloatConfig cfg;
public:
  FPMult(const UFloatConfig& cfg):
    firp::Module<FPMult>(
      "FPMult",
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