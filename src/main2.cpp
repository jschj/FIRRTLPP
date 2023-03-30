#include <iostream>
#include "firp.hpp"
#include "firpQueue.hpp"
#include "AXIStream.hpp"

using namespace firp;

class ExternalQueue : public ExternalModule<ExternalQueue> {
public:
  ExternalQueue():
    ExternalModule<ExternalQueue>(
      "SomeQueue",
      {
        Port("enq", true, readyValidType(uintType(32))),
        Port("deq", false, readyValidType(uintType(32)))
      }
    ) {}
};

class MyTop : public Module<MyTop> {
public:
  MyTop():
    Module<MyTop>(
      "MyTop",
      {

      }
    ) {}

  void body() {
    ExternalQueue q1;
    ExternalQueue q2;

    q1.io("enq")("valid") <<= io("rst");
    q1.io("enq")("bits") <<= cons(0, uintType(32));
    q1.io("deq")("ready") <<= io("rst");

    q2.io("enq")("valid") <<= io("rst");
    q2.io("enq")("bits") <<= cons(0, uintType(32));
    q2.io("deq")("ready") <<= io("rst");
  }
};

int main(int argc, const char **argv) {
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());
  assert(context->getOrLoadDialect<circt::sv::SVDialect>());

  using namespace ::firp;
  using namespace ::circt::firrtl;
  using namespace ::mlir;

  initFirpContext(context.get(), "MyTop");

  //auto myQueue = MyQueue(uintType(32), 5);
  //auto counter = Counter(123);
  //auto firpQueue = FirpQueue(uintType(32), 5);
  //firpQueue.makeTop();

  MyTop myTop;
  myTop.makeTop();

  firpContext()->finish();
  firpContext()->dump();

  return 0;
}