//===- LegalizeForLLVMExport.cpp - Prepare ArmSVE for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arm_sve;

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");

    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

using SdotOpLowering = OneToOneConvertToLLVMPattern<SdotOp, SdotIntrOp>;
using SmmlaOpLowering = OneToOneConvertToLLVMPattern<SmmlaOp, SmmlaIntrOp>;
using UdotOpLowering = OneToOneConvertToLLVMPattern<UdotOp, UdotIntrOp>;
using UmmlaOpLowering = OneToOneConvertToLLVMPattern<UmmlaOp, UmmlaIntrOp>;

/// Populate the given list with patterns that convert from ArmSVE to LLVM.
void mlir::populateArmSVELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Populate conversion patterns

  // clang-format off
  patterns.add<ForwardOperands<func::CallOp>,
               ForwardOperands<func::CallIndirectOp>,
               ForwardOperands<func::ReturnOp>>(converter,
                                          &converter.getContext());
  patterns.add<SdotOpLowering,
               SmmlaOpLowering,
               UdotOpLowering,
               UmmlaOpLowering>(converter);
  // clang-format on
}

void mlir::configureArmSVELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<SdotIntrOp,
                    SmmlaIntrOp,
                    UdotIntrOp,
                    UmmlaIntrOp>();
  target.addIllegalOp<SdotOp,
                      SmmlaOp,
                      UdotOp,
                      UmmlaOp>();
  // clang-format on
}
