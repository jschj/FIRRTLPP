
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
