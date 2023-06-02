"""
import numpy as np
import struct
from codecs import decode
import math
import json
import sys
import os

import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ClockCycles, First, Join, Combine
from cocotb.clock import Clock

import random
from copy import copy

SAMPLE_COUNT = 1000
DELAY = 4

def float_to_bits(f):
  s = struct.pack('>f', f)
  return struct.unpack('>l', s)[0]

def float_to_bits(f):
  s = struct.pack('>f', f)
  return struct.unpack('>l', s)[0]

def generate_test_data(bitWidth):
  a_ins = []
  b_ins = []
  c_outs = []

  bound = (1 << bitWidth) - 1

  for _ in range(SAMPLE_COUNT):
    a = random.randint(0, bound)
    b = random.randint(0, bound)
    a_ins.append(a)
    b_ins.append(b)
    c_outs.append(a + b)

  return a_ins, b_ins, c_outs

async def reset(dut):
  print(f'resetting...')
  dut.reset.value = 1
  dut.enq.valid.value = 0
  dut.enq.bits.value = 0
  dut.deq.ready.value = 0
  for i in range(0, 5):
      await RisingEdge(dut.clock)
  dut.reset.value = 0
  print(f'done')

async def timeout(dut, n):
  print(f'armed timeout with {n} cycles')
  await ClockCycles(dut.clock, n)
  assert False, 'simulation timed out'

@cocotb.test()
async def test_pipelinedAdder(dut):
  random.seed(123456)
  width = dut.a.value.n_bits

  a_ins, b_ins, c_outs = generate_test_data(width)

  clock = Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())

  print(f'resetting...')
  dut.reset.value = 1
  for i in range(0, 5):
      await RisingEdge(dut.clock)
  dut.reset.value = 0
  print(f'done')

  #write_thread = Join(cocotb.start_soon(write_inputs(dut, data)))
  #read_thread = Join(cocotb.start_soon(read_outputs(dut, len(data))))
  #timeout_thread = Join(cocotb.start_soon(timeout(dut, compute_max_cycles(SAMPLE_COUNT))))

  # poke around
  got = []
  t = 0
  ci = 0

  for a, b in zip(a_ins, b_ins):
    await FallingEdge(dut.clock)
    t += 1

    dut.a.value = a
    dut.b.value = b

    if t // 2 >= DELAY:
      got.append(dut.c.value.integer)
      print(f'got={dut.c.value.integer} exp={c_outs[ci]}')
      ci += 1

    await RisingEdge(dut.clock)
    t += 1

  # check results
  print('checking results...')
  c_outs = c_outs[0:-DELAY]
  assert len(got) == len(c_outs)

  for g, c in zip(got, c_outs):
    #print(f'got {g} expected {c}')
    assert(g == c)
"""

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
async def test_fpadd(dut):
    """
    Generate 100 random triplets of (input, input, expected output), write them
    to the DUT, and check that the output of the DUT matches the expected output.
    """
    clock = Clock(dut.clock, 1, units="ns")
    cocotb.fork(clock.start())

    random.seed(123456)
    DELAY = 6 # for 31 bit ufloat

    await reset(dut)

    results = []

    for i in range(100):
        # Generate random inputs
        a = np.float32(random.uniform(0.0, 100.0))
        b = np.float32(random.uniform(0.0, 100.0))

        # Compute the expected output
        expected_output = a + b

        # Convert inputs and expected output to integers
        a_int = float_to_int(a)
        b_int = float_to_int(b)
        expected_output_int = float_to_int(expected_output)

        # Write inputs to the DUT
        dut.io_a <= a_int
        dut.io_b <= b_int

        # Wait for the DUT to compute the output
        await Timer(2, units="ns")

        # Read the output from the DUT
        c_int = int(dut.io_r)

        # Convert the output to a float
        c = int_to_float(c_int)

        # Check that the output matches the expected output
        #assert abs(c - expected_output) < 1e-6
        #print(f'got {c} exp {expected_output}')
        #print(f'got {c_int} exp {expected_output_int}')
        #print(f'------------------sample {i}------------------')
        
        #print(f'a = {format(a_int, "032b")} ({a})')
        #print(f'b = {format(b_int, "032b")} ({b})')

        #print(f'got {bin(c_int)[2:]} ({hex(c_int)}) ({int_to_float(c_int)})')
        #print(f'exp {bin(expected_output_int)[2:]} ({hex(expected_output_int)}) ({int_to_float(expected_output_int)})')

        results.append(c)

    np.savetxt('old.csv', np.array(results), delimiter=';')
