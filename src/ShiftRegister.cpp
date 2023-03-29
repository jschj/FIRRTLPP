#include "ShiftRegister.hpp"


namespace firp {

void ShiftRegister::body(FIRRTLBaseType elementType, size_t depth) {
  std::vector<Reg> regs;

  for (size_t i = 0; i < depth; ++i) {
    regs.push_back(elementType);

    if (i >= 1)
      regs[i].write(regs[i - 1].read());
  }

  if (depth >= 1) {
    regs.front().write(io("in"));
    io("out") <<= regs.back().read();
  } else {
    io("out") <<= io("in");
  }
}

}