#include "firrtlpp.hpp"

namespace firrtlpp {

PrimitiveBuilder *getPrimitiveBuilder() {
  return prim.get();
}

void initPrimitiveBuilder(MLIRContext *ctxt, const std::string& topModule) {
  prim = std::make_unique<PrimitiveBuilder>(ctxt, topModule);
}

Value constant(int64_t value, uint32_t bitWidth) {
  return prim->builder.create<ConstantOp>(
    prim->builder.getUnknownLoc(),
    IntType::get(prim->context, false, bitWidth),
    ::llvm::APInt(bitWidth, value)
  ).getResult();
}

ExpressionWrapper ExpressionWrapper::operator~() const {
  return ExpressionWrapper::make<UnaryExpression>(*this, Expression::Operation::OP_NEG);
}

ExpressionWrapper ExpressionWrapper::operator&(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_AND);
}

ExpressionWrapper ExpressionWrapper::operator|(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_OR);
}

ExpressionWrapper ExpressionWrapper::operator+(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_ADD);
}

ExpressionWrapper ExpressionWrapper::operator-(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_SUB);
}

ExpressionWrapper ExpressionWrapper::operator>(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_GT);
}

ExpressionWrapper ExpressionWrapper::operator>=(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_GEQ);
}

ExpressionWrapper ExpressionWrapper::operator<(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_LT);
}

ExpressionWrapper ExpressionWrapper::operator<=(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_LEQ);
}

ExpressionWrapper ExpressionWrapper::operator==(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_EQ);
}

ExpressionWrapper ExpressionWrapper::operator!=(ExpressionWrapper b) const {
  return ExpressionWrapper::make<BinaryExpression>(*this, b, Expression::Operation::OP_NEQ);
}

ExpressionWrapper ExpressionWrapper::operator()(const std::string& fieldName) const {
  return ExpressionWrapper::make<FieldExpression>(*this, fieldName);
}

ExpressionWrapper ExpressionWrapper::operator()(size_t hi, size_t lo) const {
  return ExpressionWrapper::make<BitsExpression>(*this, hi, lo);
}

ExpressionWrapper ExpressionWrapper::operator()(size_t bitIndex) const {
  return ExpressionWrapper::make<BitsExpression>(*this, bitIndex, bitIndex);
}

}
