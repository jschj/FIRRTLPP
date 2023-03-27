#pragma once

#include "firp.hpp"


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
  AXIStreamReceiver(const AXIStreamConfig& config):
    Module<AXIStreamReceiver>(
      "AXIStreamReceiver",
      {
        Port("AXIS", true, AXIStreamBundleType(config)),
        Port("deq", false, readyValidType(withLast(uintType(config.dataBits))))
      },
      config
    ) {

  }

  void body(const AXIStreamConfig& config);
};

class AXIStreamSender : public Module<AXIStreamSender> {
public:
  AXIStreamSender(const AXIStreamConfig& config):
    Module<AXIStreamSender>(
      "AXIStreamSender",
      {
        Port("AXIS", false, AXIStreamBundleType(config)),
        Port("enq", true, readyValidType(withLast(uintType(config.dataBits))))
      },
      config
    ) {

  }

  void body(const AXIStreamConfig& config);
};

}