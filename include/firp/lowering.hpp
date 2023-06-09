#pragma once

#include "firp.hpp"

#include <filesystem>


namespace firp {

mlir::LogicalResult lowerFirrtlToHw();
mlir::LogicalResult exportVerilog(const std::string& directory);
mlir::LogicalResult setNewTopName(ModuleOp root, const std::string& newTopName);
mlir::OwningOpRef<mlir::ModuleOp> importFIRFile(const std::filesystem::path& path);

}