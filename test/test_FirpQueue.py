
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
# force some congestion
ENQ_PROB = 0.8
DEQ_PROB = 0.1

def compute_max_cycles(n):
  return int(n / min(ENQ_PROB, DEQ_PROB)) * 2

def generate_test_data(n):
  return [random.randint(0, 100) for _ in range(n)]

async def write_inputs(dut, data):
  to_enqueue = copy(data)

  while len(to_enqueue) > 0:
    x = to_enqueue[-1]

    await FallingEdge(dut.clk)

    should_enq = random.random() <= ENQ_PROB
    dut.enq_valid.value = should_enq

    if not should_enq:
      continue

    dut.enq_bits.value = x

    if dut.enq_ready.value:
      to_enqueue = to_enqueue[0:-1]

  await FallingEdge(dut.clk)
  dut.enq_valid.value = False

async def read_outputs(dut, n):
  got = []

  while len(got) != n:
    await FallingEdge(dut.clk)

    should_deq = random.random() <= DEQ_PROB
    dut.deq_ready.value = should_deq

    if not should_deq:
      continue

    await RisingEdge(dut.clk)

    if dut.deq_valid.value:
      assert dut.deq_valid.value and should_deq
      x = dut.deq_bits.value
      print(f'READ: {x.binstr}')
      got.append(int(x))

  await FallingEdge(dut.clk)
  dut.deq_ready = False
  got = list(reversed(got))
  return got

async def reset(dut):
  print(f'resetting...')
  dut.rst.value = 1
  dut.enq.valid.value = 0
  dut.enq.bits.value = 0
  dut.deq.ready.value = 0
  for i in range(0, 5):
      await RisingEdge(dut.clk)
  dut.rst.value = 0
  print(f'done')

async def timeout(dut, n):
  print(f'armed timeout with {n} cycles')
  await ClockCycles(dut.clk, n)
  assert False, 'simulation timed out'

@cocotb.test()
async def test_FirpQueue(dut):
  random.seed(123456)
  data = generate_test_data(SAMPLE_COUNT)

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  print(f'resetting...')
  dut.rst.value = 1
  dut.enq_valid.value = 0
  dut.enq_bits.value = 0
  dut.deq_ready.value = 0
  for i in range(0, 5):
      await RisingEdge(dut.clk)
  dut.rst.value = 0
  print(f'done')

  write_thread = Join(cocotb.start_soon(write_inputs(dut, data)))
  read_thread = Join(cocotb.start_soon(read_outputs(dut, len(data))))
  timeout_thread = Join(cocotb.start_soon(timeout(dut, compute_max_cycles(SAMPLE_COUNT))))

  await First(
    Combine(write_thread, read_thread),
    timeout_thread
  )

  results = read_thread.retval
  assert(len(data) == len(results))

  for got, expected in zip(results, data):
    print(f'got={got} expected={expected}')
    assert(got == expected)
