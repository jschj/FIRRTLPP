#include <firp/AXIStreamConverter.hpp>


namespace firp::axis {

void MultiRingBuffer::body() {
  uint32_t inWidth = inByteWidth * 8;
  uint32_t outWidth = outByteWidth * 8;

  uint32_t ptrWidth = clog2(slotCount);

  auto buf = Reg(vectorType(uintType(8), slotCount), "buf");
  auto inPtr = regInit(uval(1, ptrWidth), "inPtr");
  auto outPtr = regInit(uval(0, ptrWidth), "outPtr");

  auto nextPtr = [&](FValue ptr, FValue inc){
    bool isPow2 = (slotCount & (slotCount - 1)) == 0;
    assert(isPow2 && "slotCount must be a power of 2");
    // force the correct bit width
    return ((ptr + inc) & uval(slotCount - 1))(ptrWidth - 1, 0);
  };

  auto wouldNotOvertake = [&](FValue ptr1, FValue inc, FValue ptr2){
    // assume that inc has reasonable values
    auto sum = wireInit(cat({uval(0), ptr1}) + cat({uval(0), inc}), "sum").read();
    auto wouldWrap = wireInit(sum >= uval(slotCount), "wouldWrap").read();

    // [0][1][2][3][4][5][6][7]
    return mux(
      wouldWrap,
      ptr1 > ptr2 & nextPtr(ptr1, inc) < ptr2,
      ptr1 > ptr2 | ptr1 + inc < ptr2
    );
  };

  // enq logic
  io("enq")("ready") <<= wouldNotOvertake(inPtr, uval(inByteWidth, ptrWidth), outPtr);

  when(doesFire(io("enq")), [&]{
    for (uint32_t i = 0; i < inByteWidth; ++i)
      buf.read()[nextPtr(inPtr, uval(i))] <<= io("enq")("bits")(i * 8 + 7, i * 8);

    inPtr <<= nextPtr(inPtr, uval(inByteWidth, ptrWidth));
  });

  // deq logic
  io("deq")("valid") <<= wouldNotOvertake(outPtr, uval(outByteWidth, ptrWidth), inPtr);

  when(doesFire(io("deq")), [&](){
    std::vector<FValue> elementValues;  

    for (uint32_t i = 0; i < outByteWidth; ++i)
      elementValues.push_back(buf.read()[nextPtr(outPtr, uval(i + 1))]);

    io("deq")("bits") <<= cat(elementValues);

    outPtr <<= nextPtr(outPtr, uval(outByteWidth));
  }).otherwise([&](){
    std::vector<FValue> elementValues(outByteWidth, uval(0, 8));
    io("deq")("bits") <<= cat(elementValues);
  });
}

void AXIStreamConverter::body() {
  uint32_t inByteWidth = slaveConfig.dataBits / 8;
  uint32_t outByteWidth = masterConfig.dataBits / 8;
  uint32_t slotCount = 1 << clog2(slaveConfig.dataBits + masterConfig.dataBits); 

  auto buf = MultiRingBuffer(inByteWidth, outByteWidth, slotCount);

  buf.io("enq")("bits") <<= io("AXIS_slave")("TDATA");
  buf.io("enq")("valid") <<= io("AXIS_slave")("TVALID");
  io("AXIS_slave")("TREADY") <<= buf.io("enq")("ready");

  buf.io("deq")("ready") <<= io("AXIS_master")("TREADY");
  io("AXIS_master")("TDATA") <<= buf.io("deq")("bits");
  io("AXIS_master")("TVALID") <<= buf.io("deq")("valid");

  // To correctly tag the last word as last the following conditions must hold:
  // 1. Last was received from an input before.
  // 2. The word boundaries must align.
  auto inByteCount = regInit(uval(0, 16), "inByteCount");
  auto outByteCount = regInit(uval(0, 16), "outByteCount");
  auto wouldOutByteCount = Wire(uintType(16), "wouldOutByteCount");
  auto receivedLast = regInit(uval(0, 1), "receivedLast");

  when(doesFire(buf.io("enq")), [&](){
    inByteCount <<= inByteCount.read() + uval(inByteWidth);
    receivedLast <<= receivedLast.read() | io("AXIS_slave")("TLAST");
  });

  when(doesFire(buf.io("deq")), [&](){
    outByteCount <<= wouldOutByteCount;
  });

  wouldOutByteCount <<= outByteCount.read() + uval(outByteWidth);
  io("AXIS_master")("TLAST") <<= receivedLast.read() & (inByteCount.read() == wouldOutByteCount.read());

  // STRB, KEEP, USER, DEST, ID are not supported and are set to default values
  io("AXIS_master")("TSTRB") <<= uval((1 << (masterConfig.dataBits / 8)) - 1, masterConfig.dataBits / 8);
  io("AXIS_master")("TKEEP") <<= uval(0, masterConfig.dataBits / 8);
  io("AXIS_master")("TUSER") <<= uval(0, masterConfig.userBits);
  io("AXIS_master")("TDEST") <<= uval(0, masterConfig.destBits);
  io("AXIS_master")("TID") <<= uval(0, masterConfig.idBits);
}

}