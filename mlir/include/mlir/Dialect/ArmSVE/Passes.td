//===-- Passes.td - ArmSVE pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSVE_PASSES
#define MLIR_DIALECT_ARMSVE_PASSES

include "mlir/Pass/PassBase.td"

def ConvertFixedLengthToVLALoop :
                FunctionPass<"arm-sve-fixed-length-to-vla-loop"> {
  let summary = "Convert loops over fixed-length vectors to VLA loops";
  let description = [{
    Convert loops over fixed-length vector operations with the `VLAConvertible`
    attribute to vector-length-agnostic loops over scalable vector operations.
  }];
  let constructor = "mlir::createConvertFixedLengthToVLALoopPass()";
  let dependentDialects = ["arm_sve::ArmSVEDialect",
                           "StandardOpsDialect",
//                           "memref::MemRefDialect",
                           "scf::SCFDialect"];//, "vector::VectorDialect",
}

#endif // MLIR_DIALECT_ARMSVE_PASSES
