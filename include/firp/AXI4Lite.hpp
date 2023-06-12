#pragma once

#include "firp.hpp"


namespace axi4lite {

struct AXI4LiteConfig {
  uint32_t addrBits = 32;
  uint32_t dataBits = 32;
  // these are all fixed by the standard
  uint32_t protBits  = 3;
  uint32_t respBits  = 2;
};

firp::BundleType axi4LiteAddressChannelType(const AXI4LiteConfig& config);
firp::BundleType axi4LiteDataChannelType(const AXI4LiteConfig& config);
firp::BundleType axi4LiteWriteResponseChannelType(const AXI4LiteConfig& config);

firp::BundleType axi4LiteType(const AXI4LiteConfig& config);

std::vector<firp::FValue> axi4LiteRegisterFile(const AXI4LiteConfig& cfg, const std::vector<std::string>& registers, firp::FValue axi4LiteSlave);

}

// hashing support
namespace llvm {

template <>
struct DenseMapInfo<axi4lite::AXI4LiteConfig> {

  typedef std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> tuple_t;
private:
  static tuple_t toTuple(const axi4lite::AXI4LiteConfig& val) {
    return std::make_tuple(val.addrBits, val.dataBits, val.protBits, val.respBits);
  }

  static axi4lite::AXI4LiteConfig fromTuple(const tuple_t& val) {
    return axi4lite::AXI4LiteConfig {
      .addrBits = std::get<0>(val),
      .dataBits = std::get<1>(val),
      .protBits = std::get<2>(val),
      .respBits = std::get<3>(val)
    };
  }
public:
  static inline axi4lite::AXI4LiteConfig getEmptyKey() { return fromTuple(DenseMapInfo<tuple_t>::getEmptyKey()); }
  static inline axi4lite::AXI4LiteConfig getTombstoneKey() { return fromTuple(DenseMapInfo<tuple_t>::getTombstoneKey()); }

  static unsigned getHashValue(const axi4lite::AXI4LiteConfig& Val) {
    return DenseMapInfo<tuple_t>::getHashValue(toTuple(Val));
  }

  static bool isEqual(const axi4lite::AXI4LiteConfig& LHS,
                      const axi4lite::AXI4LiteConfig& RHS) {
    return DenseMapInfo<tuple_t>::isEqual(toTuple(LHS), toTuple(RHS));
  }
};

}