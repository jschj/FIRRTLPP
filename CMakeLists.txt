cmake_minimum_required(VERSION 3.13.4 FATAL_ERROR)

project(firttlpp VERSION 0.2.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# library

add_library(firp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/firp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AXIStream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ShiftRegister.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ufloat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lowering.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dspMult.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pipelinedAdder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ESI.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AXI4Lite.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AXI4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AXIStreamConverter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/timings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/queue.cpp
)
target_include_directories(firp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/
)
target_link_libraries(firp PRIVATE ${LIBS} CIRCTHW CIRCTSeq CIRCTFIRRTL)

# standalone executable

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

find_package(CIRCT REQUIRED CONFIG)
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif ()

if (POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif ()

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})



#add_executable(firrtlpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/main2.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/firp.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/AXIStream.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/ShiftRegister.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/ufloat.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/lowering.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/dspMult.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/pipelinedAdder.cpp
#)
#
#target_include_directories(firrtlpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
#
#target_link_libraries(firrtlpp PRIVATE ${LIBS}
#  CIRCTHW
#  CIRCTSeq
#  CIRCTFIRRTL
#  CIRCTFIRRTLToHW
#  CIRCTFIRRTLTransforms
#  CIRCTSeqTransforms
#  CIRCTSVTransforms
#  CIRCTExportVerilog
#)



add_executable(ufloat
  ${CMAKE_CURRENT_SOURCE_DIR}/src/firp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AXIStream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ShiftRegister.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ufloat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lowering.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dspMult.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pipelinedAdder.cpp
)

target_include_directories(ufloat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)

target_link_libraries(ufloat PRIVATE ${LIBS}
  CIRCTHW
  CIRCTSeq
  CIRCTFIRRTL
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTExportVerilog
  CIRCTImportFIRFile
)

add_executable(lower-firrtl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lowerFirrtl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/firp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lowering.cpp
)
target_include_directories(lower-firrtl PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/
)
target_link_libraries(lower-firrtl PRIVATE ${LIBS}
  CIRCTHW
  CIRCTSeq
  CIRCTFIRRTL
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTExportVerilog
  CIRCTImportFIRFile
)
