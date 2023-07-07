#include <firp/AXIStream.hpp>

#include <firp/firpQueue.hpp>


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
    {"last", false, bitType()},
    {"bits", false, type}
  });
}

AXIStreamReceiver::AXIStreamReceiver(const AXIStreamConfig& config):
  Module<AXIStreamReceiver>(
    "AXIStreamReceiver",
    {
      Port("AXIS", true, AXIStreamBundleType(config)),
      Port("deq", false, readyValidType(withLast(uintType(config.dataBits))))
    },
    config.dataBits, config.userBits, config.destBits, config.idBits
  ), config(config) { build(); }

AXIStreamSender::AXIStreamSender(const AXIStreamConfig& config):
  Module<AXIStreamSender>(
    "AXIStreamSender",
    {
      Port("AXIS", false, AXIStreamBundleType(config)),
      Port("enq", true, readyValidType(withLast(uintType(config.dataBits))))
    },
    config.dataBits, config.userBits, config.destBits, config.idBits
  ), config(config) { build(); }

void AXIStreamReceiver::body() {
  auto elType = withLast(uintType(config.dataBits));
  /*
  auto queue = FirpQueue(elType, 8);
  
  queue.io("enq")("valid") <<= io("AXIS")("TVALID");
  queue.io("enq")("bits")("bits") <<= io("AXIS")("TDATA");
  
  queue.io("enq")("bits")("last") <<= io("AXIS")("TLAST");
  io("AXIS")("TREADY") <<= queue.io("enq")("ready");

  io("deq") <<= queue.io("deq");*/

  auto dataQueue = FirpQueue(uintType(config.dataBits), 8);
  auto lastQueue = FirpQueue(uintType(1), 8);

  dataQueue.io("enq")("valid") <<= io("AXIS")("TVALID");
  dataQueue.io("enq")("bits") <<= io("AXIS")("TDATA");

  lastQueue.io("enq")("valid") <<= io("AXIS")("TVALID");
  lastQueue.io("enq")("bits") <<= io("AXIS")("TLAST");

  io("AXIS")("TREADY") <<= dataQueue.io("enq")("ready") & lastQueue.io("enq")("ready");

  io("deq")("valid") <<= dataQueue.io("deq")("valid") & lastQueue.io("deq")("valid");
  io("deq")("bits")("bits") <<= dataQueue.io("deq")("bits");
  io("deq")("bits")("last") <<= lastQueue.io("deq")("bits");

  dataQueue.io("deq")("ready") <<= io("deq")("ready");
  lastQueue.io("deq")("ready") <<= io("deq")("ready");
}

void AXIStreamSender::body() {
  auto elType = withLast(uintType(config.dataBits));
  auto queue = FirpQueue(elType, 7);

  queue.io("enq") <<= io("enq");

  io("AXIS")("TVALID") <<= queue.io("deq")("valid");
  io("AXIS")("TLAST") <<= queue.io("deq")("bits")("last");
  io("AXIS")("TDATA") <<= queue.io("deq")("bits")("bits");
  queue.io("deq")("ready") <<= io("AXIS")("TREADY");

  auto ones = ~uval(0, config.dataBytes());

  io("AXIS")("TSTRB") <<= ones;
  io("AXIS")("TKEEP") <<= ones;
  io("AXIS")("TUSER") <<= cons(0, uintType(config.userBits));
  io("AXIS")("TDEST") <<= cons(0, uintType(config.destBits));
  io("AXIS")("TID") <<= cons(0, uintType(config.idBits));
}

void AXIStreamTest::body() {
  auto receiver = AXIStreamReceiver(config);
  receiver.io("AXIS") <<= io("SLAVE");

  auto sender = AXIStreamSender(config);
  io("MASTER") <<= sender.io("AXIS");

  auto queue = FirpQueue(withLast(uintType(config.dataBits)), 8);
  queue.io("enq") <<= receiver.io("deq");
  sender.io("enq") <<= queue.io("deq");
}

}

namespace firp {

//template <>
//llvm::hash_code compute_hash(const firp::axis::AXIStreamConfig& config) {
//  return llvm::hash_combine(
//    config.dataBits,
//    config.userBits,
//    config.destBits,
//    config.idBits
//  );
//}

}