#include <firp/lowering.hpp>


static void printUsage() {
  llvm::outs() << "Usage: lower-firrtl <fir file> <top module name>\n";
}

int main(int argc, const char **argv) {
  if (argc <= 2) {
    printUsage();
    return 1;
  }

  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());
  assert(context->getOrLoadDialect<circt::sv::SVDialect>());

  std::string topName = argv[2];
  firp::createFirpContext(context.get(), topName);

  llvm::outs() << "Importing FIRRTL file " << argv[1] << "\n";
  mlir::ModuleOp mod = firp::importFIRFile(argv[1]);

  if (!mod) {
    llvm::outs() << "Failed to import FIRRTL file\n";
    return 1;
  }

  llvm::outs() << "Imported file\n";

  circt::firrtl::FModuleOp fmodOp;
  circt::firrtl::CircuitOp circtOp;

  mod.walk([&](circt::firrtl::FModuleOp op){
    if (op.getName() == topName)
      fmodOp = op;
  });

  mod.walk([&](circt::firrtl::CircuitOp op){
    circtOp = op;
  });

  if (!fmodOp || !circtOp) {
    llvm::outs() << "Failed to find ops\n";
    return 1;
  }

  firp::attachFirpContext(circtOp);
  if (mlir::failed(firp::firpContext()->finish())) {
    llvm::outs() << "Failed to finish firp context\n";
    return 1;
  }

  circtOp.dump();

  llvm::outs() << "Beginning lowering...\n";
  if (mlir::failed(firp::lowerFirrtlToHw(true))) {
    llvm::outs() << "Lowering to HW failed\n";
    return 1;
  }
  llvm::outs() << "Successfully lowered to HW\n";

  llvm::outs() << "Exporting verilog...\n";
  if (mlir::failed(firp::exportVerilog("."))) {
    llvm::outs() << "Failed to export verilog\n";
    return 1;
  }
  llvm::outs() << "Successfully exported verilog\n";

  return 0;
}