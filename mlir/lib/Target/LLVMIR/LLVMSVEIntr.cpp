//===- LLVMSVEIntr.cpp - Convert MLIR LLVM dialect to LLVM intrinsics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM and SVE dialects and
// LLVM IR with SVE intrinsics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMSVEDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;

namespace {
class LLVMSVEModuleTranslation : public LLVM::ModuleTranslation {
  friend LLVM::ModuleTranslation;

public:
  using LLVM::ModuleTranslation::ModuleTranslation;

protected:
  LogicalResult convertOperation(Operation &opInst,
                                 llvm::IRBuilder<> &builder) override {
#include "mlir/Dialect/LLVMIR/LLVMSVEConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};

std::unique_ptr<llvm::Module>
translateLLVMSVEModuleToLLVMIR(Operation *m, llvm::LLVMContext &llvmContext,
                               StringRef name) {
  return LLVM::ModuleTranslation::translateModule<LLVMSVEModuleTranslation>(
      m, llvmContext, name);
}
} // end namespace

namespace mlir {
void registerSVEToLLVMIRTranslation() {
  TranslateFromMLIRRegistration reg(
      "sve-mlir-to-llvmir",
      [](ModuleOp module, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateLLVMSVEModuleToLLVMIR(
            module, llvmContext, "LLVMDialectModule");
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<LLVM::LLVMSVEDialect, LLVM::LLVMDialect>();
      });
}
} // namespace mlir
