#include <firp/ShiftRegister.hpp>


namespace firp {

void ShiftRegister::body() {
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