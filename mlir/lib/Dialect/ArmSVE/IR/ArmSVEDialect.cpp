//===- ArmSVEDialect.cpp - MLIR ArmSVE dialect implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmSVE dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::arm_sve;

//===----------------------------------------------------------------------===//
// ScalableVector versions of general helpers for comparison ops
//===----------------------------------------------------------------------===//

/// Return the scalable vector of the same shape and containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto sVectorType = type.dyn_cast<VectorType>())
    return VectorType::get(sVectorType.getShape(), i1Type,
                           sVectorType.getNumScalableDims());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSVE/ArmSVEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSVE/ArmSVE.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/ArmSVE/ArmSVETypes.cpp.inc"

void ArmSVEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/ArmSVE/ArmSVE.cpp.inc"
      >();
}

LogicalResult DotOp::verify() {
  VectorType src1Type = src1().getType().cast<VectorType>();
  VectorType accType = acc().getType().cast<VectorType>();
  Type src1ElemType = src1Type.getElementType();
  Type accElemType = accType.getElementType();
  if (src1ElemType.isa<BFloat16Type>()) {
    if (!accElemType.isa<Float32Type>())
      return emitOpError("{acc, dst} must be 32-bit floating point type");
    if (src1Type.getNumElements() != 8 || accType.getNumElements() != 4)
      return emitOpError("{acc, dst} must be vectors of size 4, {src1, src2} "
                         "must be vectors of size 8");
  } else { // if (src1ElemType.isa<IntegerType>())
    if (!accElemType.isa<IntegerType>())
      return emitOpError("{dst, acc, src1, src2} must be of integer type");
    unsigned src1Width = src1ElemType.cast<IntegerType>().getWidth();
    unsigned accWidth = accElemType.cast<IntegerType>().getWidth();
    if (accWidth != src1Width * 4)
      return emitOpError("bit width of {src1, src2} must be four times the bit "
                         "width of the {dst, acc}");
    if (accType.getNumElements() * 4 != src1Type.getNumElements())
      return emitOpError("the dimensionality of {src1, src2} must be four times"
                         " the dimensionality of {dst, acc}");
  }
  return success();
}