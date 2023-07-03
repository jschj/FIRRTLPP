#include <firp/AXI4.hpp>


namespace axi4 {

using namespace firp;

BundleType axi4AddressChannelType(const AXI4Config& config) {
  return bundleType({
    {"READY", false, bitType()},
    {"VALID", true, bitType()},
    {"ID", true, uintType(config.idBits)},
    {"ADDR", true, uintType(config.addrBits)},
    {"LEN", true, uintType(config.lenBits)},
    {"SIZE", true, uintType(config.sizeBits)},
    {"BURST", true, uintType(config.burstBits)},
    {"LOCK", true, uintType(config.lockBits)},
    {"CACHE", true, uintType(config.cacheBits)},
    {"PROT", true, uintType(config.protBits)},
    {"QOS", true, uintType(config.qosBits)},
    {"REGION", true, uintType(config.regionBits)},
    {"USER", true, uintType(config.userBits)}
  });
}

BundleType axi4WriteChannelType(const AXI4Config& config) {
  return bundleType({
    {"READY", false, bitType()},
    {"VALID", true, bitType()},
    {"DATA", true, uintType(config.dataBits)},
    {"STRB", true, uintType(config.dataBits / 8)},
    {"LAST", true, bitType()}
  });
}

BundleType axi4ResponseChannelType(const AXI4Config& config) {
  return bundleType({
    {"READY", false, bitType()},
    {"VALID", true, bitType()},
    {"ID", true, uintType(config.idBits)},
    {"RESP", true, uintType(config.respBits)},
    {"USER", true, uintType(config.userBits)}
  });
}

BundleType axi4ReadChannelType(const AXI4Config& config) {
  return bundleType({
    {"READY", false, bitType()},
    {"VALID", true, bitType()},
    {"ID", true, uintType(config.idBits)},
    {"DATA", true, uintType(config.dataBits)},
    {"RESP", true, uintType(config.respBits)},
    {"USER", true, uintType(config.userBits)},
    {"LAST", true, bitType()}
  });
}

BundleType axi4Type(const AXI4Config& writeConfig, const AXI4Config& readConfig) {
  return bundleType({
    {"AW", true, axi4AddressChannelType(writeConfig)},
    {"W", true, axi4WriteChannelType(writeConfig)},
    {"B", false, axi4ResponseChannelType(writeConfig)},
    {"AR", true, axi4AddressChannelType(readConfig)},
    {"R", false, axi4ReadChannelType(readConfig)}
  });
}

BundleType axi4FlattenType(BundleType type) {
  std::vector<BundleType::BundleElement> newElements;

  for (const auto& el : type.getElements()) {
    for (const auto& subEl : el.type.cast<BundleType>().getElements()) {
      bool flip = el.isFlip != subEl.isFlip;

      newElements.emplace_back(
        StringAttr::get(type.getContext(), el.name.str() + subEl.name.str()),
        flip,
        subEl.type
      );
    }
  }

  return BundleType::get(type.getContext(), newElements);
}

}