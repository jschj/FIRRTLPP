#pragma once

#include "firp.hpp"

#include "llvm/ADT/Hashing.h"


namespace firp::axis {

struct AXIStreamConfig {
  uint32_t dataBits;
  uint32_t userBits = 1;
  uint32_t destBits = 1;
  uint32_t idBits = 1;

  uint32_t dataBytes() const { return dataBits / 8; }
};

BundleType AXIStreamBundleType(const AXIStreamConfig& config);
BundleType withLast(FIRRTLBaseType type);

class AXIStreamReceiver : public Module<AXIStreamReceiver> {
  AXIStreamConfig config;
public:
  AXIStreamReceiver(const AXIStreamConfig& config);
  void body();
};

class AXIStreamSender : public Module<AXIStreamSender> {
  AXIStreamConfig config;
public:
  AXIStreamSender(const AXIStreamConfig& config);
  void body();
};

class AXIStreamTest : public Module<AXIStreamTest> {
  AXIStreamConfig config;
public:
  AXIStreamTest(const AXIStreamConfig& config):
    Module<AXIStreamTest>(
      "AXIStreamTest",
      {
        Port("SLAVE", true, AXIStreamBundleType(config)),
        Port("MASTER", false, AXIStreamBundleType(config))
      },
      config.dataBits, config.userBits, config.destBits, config.idBits
    ) {}

  void body();
};

}

// hashing support
namespace llvm {

template <>
struct DenseMapInfo<firp::axis::AXIStreamConfig> {

  typedef std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> tuple_t;
private:
  static tuple_t toTuple(const firp::axis::AXIStreamConfig& val) {
    return std::make_tuple(val.dataBits, val.userBits, val.destBits, val.idBits);
  }

  static firp::axis::AXIStreamConfig fromTuple(const tuple_t& val) {
    return firp::axis::AXIStreamConfig {
      .dataBits = std::get<0>(val),
      .userBits = std::get<1>(val),
      .destBits = std::get<2>(val),
      .idBits   = std::get<3>(val)
    };
  }
public:
  static inline firp::axis::AXIStreamConfig getEmptyKey() { return fromTuple(DenseMapInfo<tuple_t>::getEmptyKey()); }
  static inline firp::axis::AXIStreamConfig getTombstoneKey() { return fromTuple(DenseMapInfo<tuple_t>::getTombstoneKey()); }

  static unsigned getHashValue(const firp::axis::AXIStreamConfig& Val) {
    return DenseMapInfo<tuple_t>::getHashValue(toTuple(Val));
  }

  static bool isEqual(const firp::axis::AXIStreamConfig& LHS,
                      const firp::axis::AXIStreamConfig& RHS) {
    return DenseMapInfo<tuple_t>::isEqual(toTuple(LHS), toTuple(RHS));
  }
};

}