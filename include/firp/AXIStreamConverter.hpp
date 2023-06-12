#pragma once

#include "AXIStream.hpp"


namespace firp::axis {

class MultiRingBuffer : public firp::Module<MultiRingBuffer> {
  uint32_t inByteWidth, outByteWidth, slotCount;
public:
  MultiRingBuffer(uint32_t inByteWidth, uint32_t outByteWidth, uint32_t slotCount):
    firp::Module<MultiRingBuffer>(
      "MultiRingBuffer",
      {
        firp::Port("enq", true, uintType(inByteWidth * 8)),
        firp::Port("deq", false, uintType(outByteWidth * 8))
      },
      inByteWidth, outByteWidth, slotCount
    ),
    inByteWidth(inByteWidth), outByteWidth(outByteWidth), slotCount(slotCount)
  { build(); }

  void body();
};

class AXIStreamConverter : public firp::Module<AXIStreamConverter> {
  AXIStreamConfig slaveConfig, masterConfig;
public:
  AXIStreamConverter(const AXIStreamConfig& slaveConfig, const AXIStreamConfig& masterConfig):
    firp::Module<AXIStreamConverter>(
      "AXIStreamConverter",
      {
        firp::Port("AXIS_slave", true, AXIStreamBundleType(slaveConfig)),
        firp::Port("AXIS_master", false, AXIStreamBundleType(masterConfig))
      },
      slaveConfig, masterConfig
    ),
    slaveConfig(slaveConfig), masterConfig(masterConfig)
  { build(); }

  void body();
};

}