#include <iostream>
#include "firp.hpp"
#include "firpQueue.hpp"
#include "AXIStream.hpp"

using namespace firp;

int main(int argc, const char **argv)
{
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());
  assert(context->getOrLoadDialect<circt::sv::SVDialect>());

  using namespace ::firp;
  using namespace ::circt::firrtl;
  using namespace ::mlir;

  initFirpContext(context.get(), "AXIStreamTest");

  //auto myQueue = MyQueue(uintType(32), 5);
  //auto counter = Counter(123);
  //auto firpQueue = FirpQueue(uintType(32), 5);
  //firpQueue.makeTop();

  using namespace axis;
  AXIStreamConfig config{
    .dataBits = 64,
    .userBits = 0,
    .destBits = 0,
    .idBits = 0
  };

  AXIStreamTest test(config);
  //AXIStreamReceiver test(config);
  test.makeTop();

  //Value v = mux(cons(1), cons(123), cons(456));

  firpContext()->dump();

  return 0;
}