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

std::vector<Port> toPrimitivePorts(const std::string& stemName, bool isInput, FIRRTLBaseType type) {
  return llvm::TypeSwitch<FIRRTLBaseType, std::vector<Port>>(type)
    .Case<IntType>([&](IntType type) -> std::vector<Port> {
      assert(type.hasWidth());

      return std::vector<Port>{
        Port(stemName, isInput, type)
      };
    })
    .Case<BundleType>([&](BundleType bundleType) -> std::vector<Port> {
      std::vector<Port> ports;

      for (const BundleType::BundleElement& el : bundleType.getElements()) {
        if (el.type.isa<IntType>())
          ports.emplace_back(stemName + "_" + el.name.str(), isInput != el.isFlip, el.type);
        else if (el.type.isa<BundleType>()) {
          std::vector<Port> primitives = toPrimitivePorts(stemName + "_" + el.name.str(), isInput != el.isFlip, el.type);
          ports.insert(ports.end(), primitives.cbegin(), primitives.cend());
        } else
          assert(false && "type not supported");
      }

      return ports;
    })
    .Default([&](FIRRTLBaseType) -> std::vector<Port> {
      assert(false && "type not supported");
    });
}

FValue ESIReceiver::io(const std::string& name) {
  if (name != "deq")
    return ExternalModule<ESIReceiver>::io(name);

  assert(false && "not implemented");

  // TODO: I think the Wire causes a "'hw.struct_extract' op used as connect destination" error.

  // We need this weird wire construction because bundlecreate does not seem to preserve flow.
  FValue wire = Wire(highType);
  connect(wire, false, "deq");
  return wire;
}

FValue ESISender::io(const std::string& name) {
  if (name != "enq")
    return ExternalModule<ESISender>::io(name);

  assert(false && "not implemented");

  // We need this weird wire construction because bundlecreate does not seem to preserve flow.
  FValue wire = Wire(highType);
  connect(wire, true, "enq");
  return wire;
}

}