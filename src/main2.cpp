#include <iostream>
#include "firp.hpp"
#include "firpQueue.hpp"
#include "AXIStream.hpp"
#include "AXI4.hpp"
#include "esi.hpp"

using namespace firp;
using namespace axi4;

class ExternalQueue : public ExternalModule<ExternalQueue> {
public:
  ExternalQueue():
    ExternalModule<ExternalQueue>(
      "SomeQueue",
      std::vector<Port>{
        Port("enq", true, readyValidType(uintType(32))),
        Port("deq", false, readyValidType(uintType(32)))
      }
    ) {}
};

class MyTop : public Module<MyTop> {
public:
  MyTop(const AXI4Config& config):
    Module<MyTop>(
      "MyTop",
      {
        {"AXI", true, axi4Type(config, config)}
      }
    ) {}

  void body() {
  }
};

int main(int argc, const char **argv) {
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());
  assert(context->getOrLoadDialect<circt::sv::SVDialect>());
  assert(context->getOrLoadDialect<circt::esi::ESIDialect>());

  using namespace ::firp;
  using namespace ::circt::firrtl;
  using namespace ::mlir;

  initFirpContext(context.get(), "MyReceiver");

  //auto myQueue = MyQueue(uintType(32), 5);
  //auto counter = Counter(123);
  //auto firpQueue = FirpQueue(uintType(32), 5);
  //firpQueue.makeTop();

  esi::esiInterface();

  //AXI4Config config;
  //MyTop myTop(config);
  //myTop.makeTop();

  //firpContext()->finish();
  firpContext()->dump();

  return 0;
}