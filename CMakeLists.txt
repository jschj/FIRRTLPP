add_library(firp
  #${CMAKE_CURRENT_SOURCE_DIR}/src/main2.cpp
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
)
target_include_directories(firp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/
)
target_link_libraries(firp PRIVATE ${LIBS}
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