//===- SVEDialect.h - MLIR Dialect for SVE ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for SVE in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SVE_SVEDIALECT_H_
#define MLIR_DIALECT_SVE_SVEDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace sve {

///===----------------------------------------------------------------------===//
///=== ScalableVectorType
///===----------------------------------------------------------------------===//

struct ScalableVectorTypeStorage : public TypeStorage {
  ScalableVectorTypeStorage(unsigned shapeSize, Type elementTy,
                    const int64_t *shapeElements)
      : shapeElements(shapeElements), shapeSize(shapeSize),
        elementType(elementTy) {}

  /// Hash key for uniquing
  using KeyTy = std::pair<ArrayRef<int64_t>, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType);
  }

  /// Construction.
  static ScalableVectorTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
    // Copy the shape into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.first);

    // Initialize the memory using placement new.
    return new (allocator.allocate<ScalableVectorTypeStorage>())
        ScalableVectorTypeStorage(shape.size(), key.second, shape.data());
  }

  ArrayRef<int64_t> getShape() const {
    return ArrayRef<int64_t>(shapeElements, shapeSize);
  }

  const int64_t *shapeElements;
  unsigned shapeSize;
  Type elementType;
};

/// Scalable vector types represent multi-dimensional SIMD vectors that will be
/// processed by a scalable vector length processor. They have a fixed
/// known constant shape with one or more dimensions.
class ScalableVectorType : public Type::TypeBase<ScalableVectorType,
    ShapedType, ScalableVectorTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new ScalableVectorType of the provided shape and element
  /// type. Assumes the arguments define a well-formed ScalableVectorType.
  static ScalableVectorType get(ArrayRef<int64_t> shape, Type elementType);

  /// Get or create a new ScalableVectorType of the provided shape and element
  /// type declared at the given, potentially unknown, location. If the
  /// ScalableVectorType defined by the arguments would be ill-formed, emit
  /// errors and return nullptr-wrapping type.
  static ScalableVectorType getChecked(ArrayRef<int64_t> shape, Type elementType,
                                Location location);

  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    ArrayRef<int64_t> shape,
                                                    Type elementType);

  static bool isValidElementType(Type t) {
    return t.isa<IntegerType, FloatType>();
  }

  ArrayRef<int64_t> getShape() const;

  Type getElementType() const;
};

} // namespace sve
} // namespace mlir

#include "mlir/Dialect/SVE/SVEDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/SVE/SVE.h.inc"

#endif // MLIR_DIALECT_SVE_SVEDIALECT_H_
