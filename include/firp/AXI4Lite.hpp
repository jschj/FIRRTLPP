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
firp::BundleType axi4LiteResponseChannelType(const AXI4LiteConfig& config);

firp::BundleType axi4LiteType(const AXI4LiteConfig& config);

class AXI4LiteRegisterFile : public firp::Module<AXI4LiteRegisterFile> {
  AXI4LiteConfig cfg;
  std::vector<std::string> registers;
public:
  AXI4LiteRegisterFile(const AXI4LiteConfig& cfg, std::initializer_list<std::string> registers):
    firp::Module<AXI4LiteRegisterFile>(
      "AXI4LiteRegisterFile",
      {
        firp::Port("AXI4LiteSlave", true, axi4LiteType(cfg))
      },
      cfg
    ),
    cfg(cfg), registers(registers) { build(); }

  void body();
};

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
      .respBits = std::get<2>(val)
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