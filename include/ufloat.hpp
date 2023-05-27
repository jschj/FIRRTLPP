#pragma once

#include "pipelinedAdder.hpp"
#include "dspMult.hpp"


namespace ufloat {

struct UFloatConfig {
  uint32_t exponentWidth;
  uint32_t mantissaWidth;

  uint32_t getWidth() const { return exponentWidth + mantissaWidth; }
  firp::BundleType getType() const;
};

firp::BundleType ufloatType(const UFloatConfig& cfg);

struct UFloat : public firp::FValue {
  // TODO: Maybe throw this away?
};

firp::FValue ufloatUnpack(firp::FValue what, const UFloatConfig& cfg);
firp::FValue ufloatPack(firp::FValue what, const UFloatConfig& cfg);

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