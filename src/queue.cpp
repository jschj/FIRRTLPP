#include <firp/queue.hpp>


namespace firp {

FValue Queue::buildRAM(FValue writeAddr, FValue writeData, FValue writeEn,
                       FValue readAddr, FValue readEn) {
  if (UIntType type = innerType.dyn_cast<UIntType>()) {
    uint32_t bitCount = type.getBitWidthOrSentinel();

    SmallVector<FValue> readDatas;

    for (uint32_t lo = 0; lo < bitCount; lo += 8) {
      uint32_t hi = std::min(lo + 8, bitCount - 1);
      uint32_t width = hi - lo + 1;
      auto ram = Memory(uintType(width), size);

      ram.writePort()("addr") <<= writeAddr;
      ram.writePort()("en") <<= writeEn;
      ram.writePort()("clk") <<= firpContext()->getClock();
      ram.writePort()("data") <<= writeData(hi, lo);
      ram.writePort()("mask") <<= ram.maskEnable();

      ram.readPort()("addr") <<= readAddr;
      ram.readPort()("en") <<= readEn;
      ram.readPort()("clk") <<= firpContext()->getClock();
      // ram.readPort()("data") is undefined when en is low
      auto readData = mux(readEn, ram.readPort()("data"), zeros(uintType(width)));
      readDatas.push_back(readData);
    }

    return cat(readDatas);
  }

  auto ram = Memory(innerType, size);

  ram.writePort()("addr") <<= writeAddr;
  ram.writePort()("en") <<= writeEn;
  ram.writePort()("clk") <<= firpContext()->getClock();
  ram.writePort()("data") <<= writeData;
  ram.writePort()("mask") <<= ram.maskEnable();

  ram.readPort()("addr") <<= readAddr;
  ram.readPort()("en") <<= readEn;
  ram.readPort()("clk") <<= firpContext()->getClock();
  // ram.readPort()("data") is undefined when en is low
  return mux(readEn, ram.readPort()("data"), zeros(innerType));
}

void Queue::body() {
  if (size == 0) {
    io("enq")("ready") <<= io("deq")("ready");
    io("deq")("valid") <<= io("enq")("valid");
    io("deq")("bits") <<= io("enq")("bits");

    return;
  } else if (size == 1) {
    Reg reg(innerType, "ram");

    auto isEmpty = regInit(uval(1), "isEmpty").read();
    auto isFull = regInit(uval(0), "isFull").read();

    io("enq")("ready") <<= ~isFull | io("deq")("ready");
    io("deq")("valid") <<= ~isEmpty | io("enq")("valid");

    auto doesEnq = wireInit(doesFire(io("enq")("valid")), "doesEnq").read();
    auto doesDeq = wireInit(doesFire(io("deq")("valid")), "doesDeq").read();

    when (doesDeq & ~doesEnq, [&]() {
      isEmpty <<= uval(1);
    });

    when (doesEnq & ~doesDeq, [&]() {
      isEmpty <<= uval(0);
    });

    when (doesEnq, [&]() {
      reg <<= io("enq")("bits");
    });

    io("deq")("bits") <<= reg.read();

    return;
  }

  uint32_t bitCount = clog2(size - 1);

  auto nextIndex = [&](FValue value) {
    auto result = mux(value == uval(size - 1), uval(0), value + uval(1));
    return result(bitCount - 1, 0); // truncate
  };

  auto enqIndex = regInit(uval(0, bitCount), "enqIndex");
  auto deqIndex = regInit(uval(0, bitCount), "deqIndex");
  auto count = regInit(uval(0, clog2(size)), "count");

  auto nextEnqIndex = wireInit(nextIndex(enqIndex), "nextEnqIndex").read();
  auto nextDeqIndex = wireInit(nextIndex(deqIndex), "nextDeqIndex").read();

  // [0][1][2][3][4]
  // There are two edge cases: When the queue is empty, and when the queue is full.
  // Then enqIndex and deqIndex are equal.

  auto isEmpty = regInit(uval(1), "isEmpty").read();
  auto isFull = regInit(uval(0), "isFull").read();

  auto doesEnq = wireInit(doesFire(io("enq")("valid")), "doesEnq").read();
  auto doesDeq = wireInit(doesFire(io("deq")("valid")), "doesDeq").read();

  auto nextCount = wireInit(
    count.read() + mux(doesEnq, uval(1), uval(0)) - mux(doesDeq, uval(1), uval(0)),
    "nextCount"
  ).read();

  count <<= nextCount;
  isEmpty <<= nextCount == uval(0);
  isFull <<= nextCount == uval(size);
  enqIndex <<= mux(doesEnq, nextEnqIndex, enqIndex.read());
  deqIndex <<= mux(doesDeq, nextDeqIndex, deqIndex.read());

  io("enq")("ready") <<= ~isFull | io("deq")("ready");
  io("deq")("valid") <<= ~isEmpty | io("enq")("valid");

  io("deq")("bits") <<= buildRAM(
    enqIndex.read(),
    io("enq")("bits"),
    doesEnq,
    deqIndex.read(),
    doesDeq
  );
}

}