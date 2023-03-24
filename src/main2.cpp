#include <iostream>
#include "firp.hpp"
#include "firpQueue.hpp"

using namespace firp;

class MyModule : public Module<MyModule> {
public:
  MyModule():
    Module<MyModule>(
      "MyModule",
      {
        Port("a", uintType(32), true),
        Port("b", uintType(32), false)
      },
      true) {}

  void body() {
    auto mem = Memory(uintType(32), 5);

    when (io("a") <= cons(3, uintType(32)), [&](){
      io("b") <<= cons(0, uintType(32));
    })
    .otherwise([&](){
      io("b") <<= cons(1, uintType(32));
    });
  }
};

class Counter : public Module<Counter> {
public:
  Counter(uint32_t maxValue):
    Module<Counter>(
      "Counter",
      {
        Port("cond", bitType(), true),
        Port("value", uintType(clog2(maxValue)), false)
      },
      true, // is top module?
      maxValue) {}

  void body(uint32_t maxValue) {
    auto reg = Reg(uintType(clog2(maxValue)));

    when (io("cond"), [&](){
      reg.write(reg.read() + cons(1));
    });
  }
};

class MyQueue : public Module<MyQueue> {
public:
  MyQueue(FIRRTLBaseType elementType, size_t depth):
    Module<MyQueue>(
      "MyQueue",
      {
        Port("enq", readyValidType(elementType), true),
        Port("deq", readyValidType(elementType), false)
      },
      true, // is top module?
      elementType, depth) {}

  void body(FIRRTLBaseType elementType, size_t depth) {
    auto mem = Memory(uintType(32), 5);

    auto enqFire = io("enq")("ready") & io("enq")("valid");
    auto deqFire = io("enq")("ready") & io("enq")("valid");

  }
};

int main(int argc, const char **argv)
{
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());

  using namespace ::firp;
  using namespace ::circt::firrtl;
  using namespace ::mlir;

  initFirpContext(context.get(), "FirpQueue");

  //auto myQueue = MyQueue(uintType(32), 5);
  //auto counter = Counter(123);
  auto firpQueue = FirpQueue(uintType(32), 5);

  //Value v = mux(cons(1), cons(123), cons(456));

  firpContext()->dump();

  return 0;
}