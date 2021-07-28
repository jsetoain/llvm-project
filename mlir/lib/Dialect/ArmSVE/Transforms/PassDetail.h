//===- PassDetail.h - ArmSVE Pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_ARMSVE_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_ARMSVE_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arm_sve {
class ArmSVEDialect;
} // end namespace arm_sve

namespace scf {
class SCFDialect;
} // end namespace scf

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace vector {
class VectorDialect;
} // end namespace vector

#define GEN_PASS_CLASSES
#include "mlir/Dialect/ArmSVE/Passes.h.inc"

} // end namespace mlir

#endif // DIALECT_ARMSVE_TRANSFORMS_PASSDETAIL_H_
