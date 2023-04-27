add_library(firp
  #${CMAKE_CURRENT_SOURCE_DIR}/src/main2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/firp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AXIStream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ShiftRegister.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ESI.cpp
)
target_include_directories(firp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/
)
target_link_libraries(firp PRIVATE ${LIBS} CIRCTHW CIRCTSeq CIRCTFIRRTL)