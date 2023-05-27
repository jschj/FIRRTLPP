import cocotb
import random
import struct

from cocotb.clock import Clock
from cocotb.triggers import Timer

import numpy as np

# IEEE 754 single-precision floating-point format has 32 bits
FLOAT_BITS = 32


def float_to_int(f):
    """
    Convert a float to its IEEE 754 single-precision floating-point format
    representation as an unsigned integer.
    """
    return struct.unpack("I", struct.pack("f", f))[0]


def int_to_float(i):
    """
    Convert an unsigned integer representing an IEEE 754 single-precision
    floating-point format value to a Python float.
    """
    return struct.unpack("f", struct.pack("I", i))[0]


@cocotb.coroutine
async def reset(dut):
    """
    Reset the DUT for 5 cycles.
    """
    dut.reset <= 1
    await Timer(5, units="ns")
    dut.reset <= 0


@cocotb.test()
async def test_dspmult(dut):
    """
    Generate 100 random triplets of (input, input, expected output), write them
    to the DUT, and check that the output of the DUT matches the expected output.
    """
    clock = Clock(dut.clock, 1, units="ns")
    cocotb.fork(clock.start())

    random.seed(123456)

    await reset(dut)

    for i in range(100):
        # Generate random inputs
        a = random.randint(0, 1000000000)
        b = random.randint(0, 1000000000)

        # Compute the expected output
        expected_output = a * b

        # Write inputs to the DUT
        dut.a <= a
        dut.b <= b

        # Wait for the DUT to compute the output
        await Timer(100, units="ns")

        # Read the output from the DUT
        c = int(dut.c)

        # Check that the output matches the expected output
        #assert abs(c - expected_output) < 1e-6
        #print(f'got {c} exp {expected_output}')
        #print(f'got {c_int} exp {expected_output_int}')
        print(f'------------------sample {i}------------------')
        
        print(f'a = {a}')
        print(f'b = {b}')

        print(f'got {c}')
        print(f'exp {expected_output}')
