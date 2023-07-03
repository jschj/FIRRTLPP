#include <firp/timings.hpp>


namespace firp {

uint32_t TimingAnalysis::getEndTimeOf(Value value) {
  if (value.isa<BlockArgument>())
    return 0;

  Operation *defOp = value.getDefiningOp();

  // instances need special handling
  if (InstanceOp instOp = dyn_cast<InstanceOp>(defOp)) {
    // all Values are results and we are only interested in the ones that are "inputs"
    uint32_t maxEndTime = 0;
    
    for (size_t portIndex = 0; portIndex < instOp.getNumResults(); ++portIndex) {
      if (instOp.getPortDirection(portIndex) != Direction::In)
        continue;

      // descend into the module and find the latest connect
      FModuleOp modOp = dyn_cast<FModuleOp>(instOp.getReferencedModule());
      Value arg = modOp.getArgument(portIndex);

      modOp.walk([&](ConnectOp connectOp){
        if (connectOp.getDest() == arg)
          maxEndTime = std::max(maxEndTime, getEndTimeOf(connectOp.getSrc()));
      });
    }

    return maxEndTime;
  }

  uint32_t maxEndTime = 0;

  for (Value arg : defOp->getOperands())
    maxEndTime = std::max(maxEndTime, getEndTimeOf(arg));

  if (isa<RegOp>(defOp) || isa<RegResetOp>(defOp))
    return maxEndTime + 1;
  else
    return maxEndTime;
}

}