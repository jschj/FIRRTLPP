#pragma once

#include "firp.hpp"


namespace {

class RegisterFile {
  FValue axi4Lite;
  std::vector<Reg> registers;
  std::unordered_map<std::string, uint32_t> indices;
public:
  RegisterFile(FValue axi4Lite, std::initializer_list<std::tuple<std::string, bool>> regs):
    axi4Lite(axi4Lite) {

    FirpQueue readCommandQueue(uintType(123), 8);
    FirpQueue readResultQueue(bundleType({
      {"error", true, bitType()},
      {"data", true, uintType(123)}
    }), 8);

    uint32_t offset = 123;

    // writeQueue
    // readQueue

    for (const auto& [name, writable] : regs) {
      // read logic

      readResultQueue.io("enq")("valid") <<= readCommandQueue.io("deq")("valid");
      readCommandQueue.io("deq")("ready") <<= readResultQueue.io("enq")("ready");
      auto readAddr = readCommandQueue.io("deq")("bits");
      auto isValidAddr = readAddr != cons(offset);

      readResultQueue.io("enq")("error") <<= 


      when (doesFire(readCommandQueue.io("deq"))) {

      }

      when (axiLite("AR")("ADDR") == cons(offset), [&](){

      });

      // write logic
      if (writable) {

      }

      offset += 123;
    }

  }

  Reg get(const std::string& name);
};

}