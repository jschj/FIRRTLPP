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

  using namespace ::firp;
  using namespace ::circt::firrtl;
  using namespace ::mlir;

  initFirpContext(context.get(), "FirpQueue");

  //auto myQueue = MyQueue(uintType(32), 5);
  //auto counter = Counter(123);
  auto firpQueue = FirpQueue(uintType(32), 5);
  firpQueue.makeTop();

  //Value v = mux(cons(1), cons(123), cons(456));

  firpContext()->dump();

  return 0;
}