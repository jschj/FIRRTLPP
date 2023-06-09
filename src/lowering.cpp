#include <firp/lowering.hpp>

#include "circt/Conversion/FIRRTLToHW.h"
#include "circt/Conversion/ExportVerilog.h"

#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"

#include "circt/Dialect/Seq/SeqPasses.h"

#include "circt/Dialect/SV/SVPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"

#include "llvm/Support/SourceMgr.h"


namespace firp {

mlir::LogicalResult lowerFirrtlToHw() {
  mlir::ModuleOp root = firpContext()->root;
  mlir::MLIRContext *ctxt = root.getContext();
  mlir::PassManager pm(ctxt);
  
  pm.addNestedPass<circt::firrtl::CircuitOp>(circt::firrtl::createInferWidthsPass());

  pm.addNestedPass<circt::firrtl::CircuitOp>(
    circt::firrtl::createLowerFIRRTLTypesPass(
      // mode
      circt::firrtl::PreserveAggregate::PreserveMode::None,
      // memory mode
      circt::firrtl::PreserveAggregate::PreserveMode::None
    )
  );

  auto &modulePM = pm.nest<circt::firrtl::CircuitOp>().nest<circt::firrtl::FModuleOp>();
  modulePM.addPass(circt::firrtl::createExpandWhensPass());

  pm.addPass(
    circt::createLowerFIRRTLToHWPass()
  );


  // export verilog doesn't know about seq.firreg

  pm.addNestedPass<circt::hw::HWModuleOp>(
    circt::seq::createSeqFIRRTLLowerToSVPass()
  );

  pm.addNestedPass<circt::hw::HWModuleOp>(
    circt::sv::createHWLegalizeModulesPass()
  );
  pm.addPass(circt::sv::createHWMemSimImplPass(
    //replSeqMem, ignoreReadEnableMem, stripMuxPragmas,
    //!isRandomEnabled(RandomKind::Mem), !isRandomEnabled(RandomKind::Reg),
    //addVivadoRAMAddressConflictSynthesisBugWorkaround
  ));

  return pm.run(root);
}

mlir::LogicalResult exportVerilog(const std::string& directory) {
  mlir::PassManager pm(firpContext()->context());
  pm.addPass(
    circt::createExportSplitVerilogPass(directory)
  );

  return pm.run(firpContext()->root);
}

mlir::LogicalResult setNewTopName(ModuleOp root, const std::string& newTopName) {
  class RewriteInstance : public OpConversionPattern<circt::firrtl::CircuitOp> {
    std::string newName;
  public:
    RewriteInstance(MLIRContext *ctxt, const std::string& newName):
      OpConversionPattern(ctxt), newName(newName) {}

    LogicalResult matchAndRewrite(circt::firrtl::CircuitOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
      CircuitOp newOp = rewriter.replaceOpWithNewOp<circt::firrtl::CircuitOp>(op, rewriter.getStringAttr(newName));
      newOp.getBody().takeBody(op.getBody());
      return mlir::success();
    }

  };

  MLIRContext *ctxt = root.getContext();

  ConversionTarget target(*ctxt);

  //target.addLegalDialect<::circt::firrtl::FirrtlDialect>();
  target.addDynamicallyLegalDialect<::circt::firrtl::FIRRTLDialect>([&](Operation *op) {
    // returns true if the op is legal
    circt::firrtl::CircuitOp circuitOp = dyn_cast<circt::firrtl::CircuitOp>(op);
    bool isLegal = !circuitOp || circuitOp.getName() == newTopName;
    return isLegal;
  });

  RewritePatternSet patterns(ctxt);
  patterns.add<RewriteInstance>(ctxt, newTopName);

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  root.dump();

  return applyPartialConversion(root, target, frozenPatterns);
}

mlir::OwningOpRef<mlir::ModuleOp> importFIRFile(const std::filesystem::path& path) {
  mlir::MLIRContext *ctxt = firpContext()->context();

  llvm::SourceMgr sourceMgr;
  std::string includedFile;
  sourceMgr.AddIncludeFile(path.string(), llvm::SMLoc(), includedFile);

  TimingScope ts;

  mlir::OwningOpRef<mlir::ModuleOp> mod = circt::firrtl::importFIRFile(sourceMgr, ctxt, ts, {});

  return mod;
}

}