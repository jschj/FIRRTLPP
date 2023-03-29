#pragma once

#include "firp.hpp"

#include "llvm/ADT/Hashing.h"


namespace firp::axis {

struct AXIStreamConfig {
  uint32_t dataBits;
  uint32_t userBits;
  uint32_t destBits;
  uint32_t idBits;

  uint32_t dataBytes() const { return dataBits / 8; }
};

BundleType AXIStreamBundleType(const AXIStreamConfig& config);
BundleType withLast(FIRRTLBaseType type);

class AXIStreamReceiver : public Module<AXIStreamReceiver> {
public:
  AXIStreamReceiver(const AXIStreamConfig& config);
  void body(const AXIStreamConfig& config);
};

class AXIStreamSender : public Module<AXIStreamSender> {
public:
  AXIStreamSender(const AXIStreamConfig& config);
  void body(const AXIStreamConfig& config);
};

class AXIStreamTest : public Module<AXIStreamTest> {
public:
  AXIStreamTest(const AXIStreamConfig& config):
    Module<AXIStreamTest>(
      "AXIStreamTest",
      {
        Port("SLAVE", true, AXIStreamBundleType(config)),
        Port("MASTER", false, AXIStreamBundleType(config))
      },
      config
    ) {}

  void body(const AXIStreamConfig& config);
};

}