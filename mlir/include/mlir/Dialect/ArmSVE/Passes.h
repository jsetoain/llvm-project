//===- Passes.h - ArmSVE pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSVE_PASSES_H_
#define MLIR_DIALECT_ARMSVE_PASSES_H_

#include "mlir/Pass/Pass.h"
//#include "mlir/Dialect/SCF/SCF.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createConvertFixedLengthToVLALoopPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/ArmSVE/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_ARMSVE_PASSES_H_
