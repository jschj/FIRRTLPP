#include <iostream>
#include "firrtlpp.hpp"
#include "queue.hpp"


int main(int argc, const char **argv)
{
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());

  using namespace ::firrtlpp;
  using namespace ::circt::firrtl;
  using namespace ::mlir;

  initPrimitiveBuilder(context.get(), "SmallQueue");

  /*
  auto valid = UInt(1, 1);
  auto ready = UInt(1, 1);
  auto count = UInt(123, 16);

  Value canEnqueue = (valid & ready & (count < lift(constant(8, 16))))->build();

  auto lower = bits(count, 3, 0);
  auto someField = field(lower, "someField");

  auto reg = Reg(UInt(32));
  reg << count + count;
  reg << count + count;
   */

  //auto queue = TestModuleA();
  auto queue = SmallQueue(UInt(32), 5);

  assert(succeeded(getPrimitiveBuilder()->root.verify()));
  getPrimitiveBuilder()->dump();

  return 0;
}