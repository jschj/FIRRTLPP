#include <firp/ESI.hpp>

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"


namespace firp::esi {

Type firrtlTypeToHWType(FIRRTLBaseType firrtlType) {
  return lowerType(firrtlType);
}

FIRRTLBaseType esiTypeToBundleType(Type esiType) {
  return FIRRTLBaseType();
}

SmallVector<Value> unpack(Value value, OpBuilder builder) {
  using namespace circt::hw;

  return TypeSwitch<Type, SmallVector<Value>>(value.getType())
    .Case<StructType>([&](StructType structType) {
      SmallVector<Value> values;

      for (auto &element : structType.getElements()) {
        Value extracted = builder.create<StructExtractOp>(
          builder.getUnknownLoc(),
          value,
          element.name
        ).getResult();

        for (auto e : unpack(extracted, builder))
          values.push_back(e);
      }

      return values;
    })
    .Case<ArrayType>([&](ArrayType arrayType) {
      SmallVector<Value> values;

      for (size_t i = 0; i < arrayType.getSize(); i++) {
        size_t bits = clog2(arrayType.getSize());

        Value index = builder.create<circt::hw::ConstantOp>(
          builder.getUnknownLoc(),
          builder.getIntegerType(bits),
          builder.getIntegerAttr(builder.getIntegerType(bits), i)
        );

        Value extracted = builder.create<ArrayGetOp>(
          builder.getUnknownLoc(),
          value,
          index
        ).getResult();

        for (auto e : unpack(extracted, builder))
          values.push_back(e);
      }

      return values;
    })
    .Case<IntegerType>([&](IntegerType integerType) {
      return SmallVector<Value>{value};
    })
    .Default([&](Type type) {
      type.dump();
      llvm_unreachable("unhandled type");
      return SmallVector<Value>{};
    });
}

Value pack(ArrayRef<Value> values, Type type, OpBuilder builder, size_t *index) {
  using namespace circt::hw;

  return TypeSwitch<Type, Value>(type)
    .Case<StructType>([&](StructType structType) {
      SmallVector<Value> packed;

      for (auto &element : structType.getElements()) {
        size_t n = 0;
        Value packedElement = pack(values.slice(*index), element.type, builder, &n);
        *index += n;
        packed.push_back(packedElement);
      }

      return builder.create<StructCreateOp>(
        builder.getUnknownLoc(),
        structType,
        packed
      ).getResult();
    })
    .Case<ArrayType>([&](ArrayType arrayType) {
      SmallVector<Value> packed;

      for (size_t i = 0; i < arrayType.getSize(); i++) {
        size_t n = 0;
        Value packedElement = pack(values.slice(*index), arrayType.getElementType(), builder, &n);
        *index += n;
        packed.push_back(packedElement);
      }

      return builder.create<ArrayCreateOp>(
        builder.getUnknownLoc(),
        arrayType,
        packed
      ).getResult();
    })
    .Case<IntegerType>([&](IntegerType integerType) {
      // get the current index first, then increment
      size_t i = (*index)++;
      return values[i];
    })
    .Default([&](Type type) {
      type.dump();
      llvm_unreachable("unhandled type");
      return Value();
    });
}

}