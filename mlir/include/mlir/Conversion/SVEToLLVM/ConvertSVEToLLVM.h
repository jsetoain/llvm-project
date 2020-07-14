//===- ConvertSVEToLLVM.h - Conversion Patterns from SVE to LLVM ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EDGE_CONVERSION_SVETOLLVM_CONVERTSVETOLLVM_H_
#define MLIR_EDGE_CONVERSION_SVETOLLVM_CONVERTSVETOLLVM_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ModuleOp;
template <typename T> class OperationPass;
class OwningRewritePatternList;

/// Collect a set of patterns to convert from the SVE dialect to LLVM.
void populateSVEToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            OwningRewritePatternList &patterns);

/// Create a pass to convert SVE operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertSVEToLLVMPass();

} // namespace mlir

#endif // MLIR_EDGE_CONVERSION_SVETOLLVM_CONVERTSVETOLLVM_H_
