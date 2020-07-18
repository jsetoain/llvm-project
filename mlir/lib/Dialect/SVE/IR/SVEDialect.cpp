//===- SVEDialect.cpp - MLIR SVE dialect implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SVE dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/SVE/SVEDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

void sve::SVEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SVE/SVE.cpp.inc"
      >();
  addTypes<ScalableVectorType>();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/SVE/SVE.cpp.inc"

namespace mlir {
namespace sve {

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

ScalableVectorType ScalableVectorType::get(ArrayRef<int64_t> shape,
                                           Type elementType) {
  return Base::get(elementType.getContext(), shape, elementType);
}

ScalableVectorType ScalableVectorType::getChecked(ArrayRef<int64_t> shape,
                                                  Type elementType,
                                                  Location location) {
  return Base::getChecked(location, shape, elementType);
}

ArrayRef<int64_t> ScalableVectorType::getShape() const {
  return getImpl()->getShape();
}

Type ScalableVectorType::getElementType() const {
  return getImpl()->elementType;
}

LogicalResult ScalableVectorType::verifyConstructionInvariants(
                                      Location loc,
                                      ArrayRef<int64_t> shape,
                                      Type elementType) {
  if (shape.empty())
    return emitError(loc,
            "scalable vector types must have at least one dimension");

  if (!isValidElementType(elementType))
    return emitError(loc, "vector elements must be int or float type");

  /// XXX: I think this is testing for '?' dimensions
  /// XXX: but message says something else
  if (any_of(shape, [](int64_t i) { return i <= 0; }))
    return emitError(loc, "vector types must have positive constant sizes");

  return success();
}

Type SVEDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  VectorType vectorTypeType;
  if (parser.parseType(vectorTypeType)) {
    parser.emitError(typeLoc, "unknown type in SVE dialect");
    return Type();
  }
  return ScalableVectorType::get(vectorTypeType.getShape(),
                                 vectorTypeType.getElementType());
}

void SVEDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
    .Case<ScalableVectorType>([&](ScalableVectorType svTy) {
      os << "vector<";
        for (int64_t dim : svTy.getShape())
          os << dim << 'x';
        os << svTy.getElementType() << '>';
    })
    .Default([](Type) { llvm_unreachable("unexpected 'sve' type kind"); });
}

}  // namespace sve
} // namespace mlir
