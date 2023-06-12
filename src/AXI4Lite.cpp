#include <firp/AXI4Lite.hpp>

#include <firp/firpQueue.hpp>


namespace axi4lite {

using namespace firp;

BundleType axi4LiteAddressChannelType(const AXI4LiteConfig& config) {
  return bundleType({
    {"VALID", false, bitType()},
    {"READY", true, bitType()},
    {"ADDR", false, uintType(config.addrBits)},
    {"PROT", false, uintType(config.protBits)}
  });
}

BundleType axi4LiteWriteChannelType(const AXI4LiteConfig& config) {
  return bundleType({
    {"VALID", false, bitType()},
    {"READY", true, bitType()},
    {"DATA", false, uintType(config.dataBits)},
    {"STRB", false, uintType(config.dataBits / 8)}
  });
}

BundleType axi4LiteWriteResponseChannelType(const AXI4LiteConfig& config) {
  return bundleType({
    {"VALID", false, bitType()},
    {"READY", true, bitType()},
    {"RESP", false, uintType(config.respBits)}
  });
}

BundleType axi4LiteReadDataChannelType(const AXI4LiteConfig& config) {
  return bundleType({
    {"VALID", false, bitType()},
    {"READY", true, bitType()},
    {"DATA", false, uintType(config.dataBits)},
    {"RESP", false, uintType(config.respBits)}
  });
}

BundleType axi4LiteType(const AXI4LiteConfig& config) {
  return bundleType({
    {"AW", false, axi4LiteAddressChannelType(config)},
    {"W", false, axi4LiteWriteChannelType(config)},
    {"B", true, axi4LiteWriteResponseChannelType(config)},
    {"AR", false, axi4LiteAddressChannelType(config)},
    {"R", true, axi4LiteReadDataChannelType(config)}
  });
}

std::vector<FValue> axi4LiteRegisterFile(const AXI4LiteConfig& cfg, const std::vector<std::string>& registers, FValue axi4LiteSlave) {
  const uint32_t QUEUE_SIZE = 1;

  // write address
  FirpQueue writeAddr(uintType(cfg.addrBits), QUEUE_SIZE);
  writeAddr.io("enq")("valid") <<= axi4LiteSlave("AW")("VALID");
  writeAddr.io("enq")("bits") <<= axi4LiteSlave("AW")("ADDR");
  axi4LiteSlave("AW")("READY") <<= writeAddr.io("deq")("ready");
  // always ready
  writeAddr.io("deq")("ready") <<= uval(1);

  // write data
  FirpQueue writeData(uintType(cfg.dataBits), QUEUE_SIZE);
  writeData.io("enq")("valid") <<= axi4LiteSlave("W")("VALID");
  writeData.io("enq")("bits") <<= axi4LiteSlave("W")("DATA");
  axi4LiteSlave("W")("READY") <<= writeData.io("deq")("ready");
  writeData.io("deq")("ready") <<= uval(1);

  // write response
  FirpQueue writeResp(uintType(cfg.respBits), QUEUE_SIZE);
  axi4LiteSlave("B")("VALID") <<= writeResp.io("deq")("valid");
  axi4LiteSlave("B")("RESP") <<= writeResp.io("deq")("bits");
  writeResp.io("deq")("ready") <<= axi4LiteSlave("B")("READY");

  // read address
  FirpQueue readAddr(uintType(cfg.addrBits), QUEUE_SIZE);
  readAddr.io("enq")("valid") <<= axi4LiteSlave("AR")("VALID");
  readAddr.io("enq")("bits") <<= axi4LiteSlave("AR")("ADDR");
  axi4LiteSlave("AR")("READY") <<= readAddr.io("deq")("ready");

  // read response
  FirpQueue readResp(uintType(cfg.dataBits), QUEUE_SIZE);
  axi4LiteSlave("R")("VALID") <<= readResp.io("deq")("valid");
  axi4LiteSlave("R")("DATA") <<= readResp.io("deq")("bits");
  axi4LiteSlave("R")("RESP") <<= uval(0, cfg.respBits);
  readResp.io("deq")("ready") <<= axi4LiteSlave("R")("READY");

  BundleType writeCommand = bundleType({
    {"addr", false, uintType(cfg.addrBits)},
    {"data", false, uintType(cfg.dataBits)}
  });

  FirpQueue writeCommandQueue(writeCommand, QUEUE_SIZE);
  writeCommandQueue.io("enq")("bits")("addr") <<= writeAddr.io("deq")("bits");
  writeCommandQueue.io("enq")("bits")("data") <<= writeData.io("deq")("bits");
  writeCommandQueue.io("enq")("valid") <<= writeAddr.io("deq")("valid") & writeData.io("deq")("valid");
  writeAddr.io("deq")("ready") <<= writeCommandQueue.io("enq")("ready");
  writeData.io("deq")("ready") <<= writeCommandQueue.io("enq")("ready");

  std::vector<Reg> regs;
  uint32_t offset = 0;

  for (const auto& name : registers) {
    auto reg = regInit(uval(0, cfg.dataBits), name);
    regs.push_back(reg);

    // writing
    auto writeAddrMatch = writeCommandQueue.io("deq")("bits")("addr") == uval(offset);
    auto writeValid = writeCommandQueue.io("deq")("valid");
    auto doWrite = wireInit(writeAddrMatch & writeValid, "doWrite_" + name);

    reg <<= mux(
      doWrite,
      writeCommandQueue.io("deq")("bits")("data"),
      reg.read()
    );

    offset += cfg.dataBits / 8;
  }

  // reading
  auto readIndex = readAddr.io("deq")("bits") / uval((cfg.dataBits / 8));
  auto readValid = readAddr.io("deq")("valid");
  auto regVector = vector(regs);
  readResp.io("enq")("bits") <<= mux(readValid, regVector[readIndex], uval(0, cfg.dataBits));
  readResp.io("enq")("valid") <<= readAddr.io("deq")("valid");
  readAddr.io("deq")("ready") <<= readResp.io("enq")("ready");

  // writing always succeeds (even if the address does not exist)
  writeResp.io("enq")("valid") <<= writeCommandQueue.io("deq")("valid");
  writeResp.io("enq")("bits") <<= uval(0, cfg.respBits);
  writeCommandQueue.io("deq")("ready") <<= writeResp.io("enq")("ready");

  // can only process read addresses when there is space for a response
  readAddr.io("deq")("ready") <<= readResp.io("enq")("ready");

  std::vector<FValue> regAccessors;
  for (auto reg : regs)
    regAccessors.push_back(reg.read());
  return regAccessors;
}

}