#include "ESI.hpp"


namespace firp::esi {

Type lowerFIRRTLType(FIRRTLBaseType type) {
  assert(type.isPassive() && "type must have no direction");

  MLIRContext *ctxt = firpContext()->context();
  OpBuilder& builder = firpContext()->builder();

  return llvm::TypeSwitch<FIRRTLBaseType, Type>(type)
    .Case<IntType>([&](IntType type) -> Type {
      assert(type.hasWidth());
      return circt::hw::IntType::get(
        builder.getI32IntegerAttr(type.getBitWidthOrSentinel())
      );
    })
    .Case<BundleType>([&](BundleType bundleType) -> Type {
      using FieldInfo = circt::hw::detail::FieldInfo;
      std::vector<FieldInfo> structFieldInfos;

      for (const auto& element : bundleType.getElements())
        structFieldInfos.push_back(FieldInfo{
          .name = element.name,
          .type = lowerFIRRTLType(element.type)
        });

      return circt::hw::StructType::get(ctxt, structFieldInfos);
    })
    .Default([&](FIRRTLBaseType) -> Type {
      assert(false && "type not supported");
    });
}

void unwrapAndConnect(circt::hw::HWModuleOp dst, circt::esi::ChannelType srcChan) {
   //srcChan.getInner();
}

}