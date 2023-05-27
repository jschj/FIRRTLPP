#include <firp/lowering.hpp>

#include "circt/Conversion/FIRRTLToHW.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/Pass/PassManager.h"


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

}