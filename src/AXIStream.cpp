#include "AXIStream.hpp"

#include "firpQueue.hpp"


namespace firp::axis {

BundleType AXIStreamBundleType(const AXIStreamConfig& config) {
  return bundleType({
    {"TVALID", false, bitType()},
    {"TREADY", true, bitType()},
    {"TDATA", false, uintType(config.dataBits)},
    {"TSTRB", false, uintType(config.dataBytes())},
    {"TKEEP", false, uintType(config.dataBytes())},
    {"TLAST", false, bitType()},
    {"TUSER", false, uintType(config.userBits)},
    {"TDEST", false, uintType(config.destBits)},
    {"TID", false, uintType(config.idBits)}
  });
}

BundleType withLast(FIRRTLBaseType type) {
  return bundleType({
    {"bits", false, type},
    {"last", false, bitType()}
  });
}

void AXIStreamReceiver::body(const AXIStreamConfig& config) {
  auto elType = withLast(uintType(config.dataBits));
  auto queue = FirpQueue(elType, 8);

  queue.io("enq")("valid") <<= io("AXIS")("TVALID");
  queue.io("enq")("bits") <<= io("AXIS")("TDATA");
  queue.io("enq")("last") <<= io("AXIS")("TLAST");
  io("AXIS")("TREADY") <<= queue.io("enq")("ready");

  io("deq") <<= queue.io("deq");
}

void AXIStreamSender::body(const AXIStreamConfig& config) {
  auto elType = withLast(uintType(config.dataBits));
  auto queue = FirpQueue(elType, 8);

  queue.io("enq") <<= io("enq");

  io("AXIS")("TVALID") <<= queue.io("deq")("valid");
  io("AXIS")("TLAST") <<= queue.io("deq")("last");
  io("AXIS")("TDATA") <<= queue.io("deq")("bits");
  queue.io("deq")("ready") <<= io("AXIS")("TREADY");

  auto ones = cons((1 << config.dataBytes()) - 1, uintType(config.dataBytes()));
  io("AXIS")("TSTRB") <<= ones;
  io("AXIS")("TKEEP") <<= ones;
  io("AXIS")("TUSER") <<= cons(0);
  io("AXIS")("TDEST") <<= cons(0);
  io("AXIS")("TID") <<= cons(0);
}

}