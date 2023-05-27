#!/bin/bash

BASE=$BASE_DIR/firrtlpp

# generate mlir
rm -rf tmp
mkdir tmp
cd tmp
$BASE/build/firrtlpp &> adder.mlir

# generate verilog
firtool --split-verilog -o . adder.mlir

# verilate




# verilator -Wall --cc tvmix_calc.v --exe tvmix_calc.cpp


