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

#include "mlir/Dialect/SVE/SVEDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

void sve::SVEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SVE/SVE.cpp.inc"
      >();
}

namespace mlir {
namespace sve {
#define GET_OP_CLASSES
#include "mlir/Dialect/SVE/SVE.cpp.inc"
}  // namespace sve
} // namespace mlir

