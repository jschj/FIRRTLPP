#!/bin/bash

PROJECT_DIR=$BASE_DIR/firrtlpp
TOP=$1

rm -rf tmp
mkdir tmp
cd tmp

# generate verilog files
$PROJECT_DIR/build/firrtlpp

# create header and implementation
verilator --trace --cc $TOP.sv
#make -C obj_dir -f Vtvmix_calc.mk

# copy driver and test and compile
cp $PROJECT_DIR/src/firpTestDriver.cpp.t obj_dir/firpTestDriver.cpp
cp $PROJECT_DIR/include/ufloatTest.hpp obj_dir/

verilator --trace --cc $TOP.sv --exe firpTestDriver.cpp \
-DVTOP=V$TOP \
-DVTOP_HEADER=\"$TOP.h\" \
-DVTOP_TEST_HEADER=\"ufloatTest.h\" \
-DVTEST_FUNC=pipelinedAdderTest
make -C obj_dir -f V$TOP.mk

# run simulation
obj_dir/V$TOP