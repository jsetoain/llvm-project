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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::arm_sve;

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
