#include <iostream>
#include "firp.hpp"
#include "firpQueue.hpp"
#include "AXIStream.hpp"
#include "AXI4.hpp"
//#include "esi.hpp"
#include "ufloat.hpp"
#include "lowering.hpp"

using namespace firp;
//using namespace axi4;
using namespace firp::axis;

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

class ExternalMemory : public ExternalModule<ExternalMemory> {
public:
  ExternalMemory():
    ExternalModule<ExternalMemory>(
      "ExternalMemory",
      {
        Port("AXI", true,
          AXIStreamBundleType(
            AXIStreamConfig {
              .dataBits = 8,
              .userBits = 8,
              .destBits = 8,
              .idBits = 8
            }
          )
        )
      }
    ) {}
};

class MyTop : public Module<MyTop> {
  FIRRTLBaseType innerType;
public:
  /*
  MyTop(const AXI4Config& config):
    Module<MyTop>(
      "MyTop",
      {
        {"AXI", true, axi4Type(config, config)}
      }
    ) {}*/

  MyTop(FIRRTLBaseType innerType):
    Module<MyTop>(
      "MyTop",
      {
        Port("enq", true, readyValidType(innerType)),
        Port("deq", false, readyValidType(innerType))
      },
      innerType
    ),
    innerType(innerType) {}

  void body() {
    // Problem: This modifies the constructables map!
    // Problem: this is invalid when building!
    //FirpQueue q(innerType, 5);
    ExternalQueue q;
    ExternalMemory mem;
    //FirpQueue q2(innerType, 6);
    q.io("enq") <<= io("enq");
    io("deq") <<= q.io("deq");
  }
};

int main(int argc, const char **argv) {
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());
  assert(context->getOrLoadDialect<circt::sv::SVDialect>());
  //assert(context->getOrLoadDialect<circt::esi::ESIDialect>());

  using namespace ::firp;
  using namespace ::circt::firrtl;
  using namespace ::mlir;

  initFirpContext(context.get(), "PipelinedAdder");

  //llvm::outs()
  //  << firpContext()->moduleBuilder->getSignatureId(123, 456) << "\n"
  //  << firpContext()->moduleBuilder->getSignatureId(StringRef("def"), StringRef("lol")) << "\n"
  //  << firpContext()->moduleBuilder->getSignatureId(StringRef("abcd"), 123, 456) << "\n"
  //  << firpContext()->moduleBuilder->getSignatureId(StringRef("abcd"), 123, 456, 1.0) << "\n"
  //  << firpContext()->moduleBuilder->getSignatureId(StringRef("abcd"), 123, 456, 1.0) << "\n";

  //auto myQueue = MyQueue(uintType(32), 5);
  //auto counter = Counter(123);
  //auto firpQueue = FirpQueue(uintType(32), 5);
  //firpQueue.makeTop();

  //AXI4Config config;
  //MyTop myTop(config);
  //myTop.makeTop();
  {
    //MyTop myTop(uintType(32));
    //myTop.makeTop();

    ufloat::PipelinedAdder adder(16, 4);
    adder.makeTop();
  }

  firpContext()->finish();
  firpContext()->dump();

  assert(succeeded(lowerFirrtlToHw()));
  assert(succeeded(exportVerilog(".")));

  return 0;
}