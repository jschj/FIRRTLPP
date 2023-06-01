#pragma once

#include "firp.hpp"

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESITypes.h"
//#include "circt/Dialect/ESI/HWArithTypes.h"


/**
 * The problem is the following:
 * Lowering FIRRTL to HW in general requires lowering bundle types to ground types.
 * However, since we want to benefit from the ESI dialect, we need to keep the structured types.
 * For this we implement some functionality that lets us translate between ESI structured types,
 * FIRRTL bundle types and lowered FIRRTL ground types.
 * Additionally, we provide some means to build endpoint/channel modules.
*/

/**
 * 
 * 
 *    firrtl.module @ESIEndpoint(clk, rst, in %enq: ..., in %deq: ...) {
 *
 *    }
 * 
 * 
 *    firrtl.module @ESITop(clk, rst) { ... }
 * 
*/

namespace firp::esi {

class TypeRelation {
  FIRRTLBaseType firrtlType; // can be structured
  FIRRTLBaseType loweredFirrtlType; // unstructured
  circt::esi::AnyType esiType;
public:
  // If bundleType if for example {last: uint1, bits: {v_0: uint8, v_1: uint8}} then it is lowered to
  // uint17 and is equivalent to the ESI type ???
  TypeRelation(FIRRTLBaseType firrtlType);
};

Type firrtlTypeToHWType(FIRRTLBaseType firrtlType);
FIRRTLBaseType esiTypeToBundleType(Type esiType);

class ESITop : public ExternalModule<ESITop> {
public:
  ESITop():
    ExternalModule<ESITop>(
      "ESITop",
      {

      }) {}
};

// This functions takes a structured type and returns a list of ground types.
SmallVector<Value> unpack(Value value, OpBuilder builder);
Value pack(ArrayRef<Value> values, Type type, OpBuilder builder, size_t *index);

}