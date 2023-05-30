#pragma once

#include "firp.hpp"


namespace firp {

mlir::LogicalResult lowerFirrtlToHw();
mlir::LogicalResult exportVerilog(const std::string& directory);
mlir::LogicalResult setNewTopName(ModuleOp root, const std::string& newTopName);

}