#pragma once

#include "firp.hpp"


namespace axi4 {

using namespace firp;

struct AXI4Config {
  uint32_t addrBits = 32;
  uint32_t dataBits = 32;
  uint32_t idBits = 1;
  uint32_t userBits = 1;
  // these are all fixed by the AXI4 standard
  uint32_t lenBits   = 8;
  uint32_t sizeBits  = 3;
  uint32_t burstBits = 2;
  uint32_t lockBits  = 1;
  uint32_t cacheBits = 4;
  uint32_t protBits  = 3;
  uint32_t qosBits   = 4;
  uint32_t respBits  = 2;
  uint32_t regionBits = 4;
};

BundleType axi4AddressChannelType(const AXI4Config& config);
BundleType axi4WriteChannelType(const AXI4Config& config);
BundleType axi4ResponseChannelType(const AXI4Config& config);
BundleType axi4ReadChannelType(const AXI4Config& config);
// If input is set to true, this is considered a slave port. Otherwise, it is a master port.
BundleType axi4Type(const AXI4Config& writeConfig, const AXI4Config& readConfig);

}

// hashing support
namespace llvm {

template <>
struct DenseMapInfo<axi4::AXI4Config> {

  typedef std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> tuple_t;
private:
  static tuple_t toTuple(const axi4::AXI4Config& val) {
    return std::make_tuple(val.addrBits, val.dataBits, val.idBits, val.userBits);
  }

  static axi4::AXI4Config fromTuple(const tuple_t& val) {
    return axi4::AXI4Config {
      .addrBits = std::get<0>(val),
      .dataBits = std::get<1>(val),
      .idBits   = std::get<2>(val),
      .userBits = std::get<3>(val)
    };
  }
public:
  static inline axi4::AXI4Config getEmptyKey() { return fromTuple(DenseMapInfo<tuple_t>::getEmptyKey()); }
  static inline axi4::AXI4Config getTombstoneKey() { return fromTuple(DenseMapInfo<tuple_t>::getTombstoneKey()); }

  static unsigned getHashValue(const axi4::AXI4Config& Val) {
    return DenseMapInfo<tuple_t>::getHashValue(toTuple(Val));
  }

  static bool isEqual(const axi4::AXI4Config& LHS,
                      const axi4::AXI4Config& RHS) {
    return DenseMapInfo<tuple_t>::isEqual(toTuple(LHS), toTuple(RHS));
  }
};

}