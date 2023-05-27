#pragma once

#include "firp.hpp"


namespace axi4 {

using namespace firp;

struct AXI4Config {
  uint32_t addrBits = 32;
  uint32_t dataBits = 32;
  uint32_t idBits = 1;
  uint32_t userBits = 0;
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
BundleType axi4Type(const AXI4Config& writeConfig, const AXI4Config& readConfig);

}