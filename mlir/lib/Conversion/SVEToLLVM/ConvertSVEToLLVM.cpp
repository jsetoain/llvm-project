//===- ConvertSVEToLLVM.cpp - Convert SVE to the LLVM dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SVEToLLVM/ConvertSVEToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/SVE/SVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::sve;

template <typename OpTy>
static Type getSrc1VectorElementType(OpTy op) {
  return op.src1().getType().template cast<ScalableVectorType>().getElementType();
}
template <typename OpTy>
static Type getSrc2VectorElementType(OpTy op) {
  return op.src2().getType().template cast<ScalableVectorType>().getElementType();
}
template <typename OpTy>
static Type getAccVectorElementType(OpTy op) {
  return op.acc().getType().template cast<ScalableVectorType>().getElementType();
}

/// TODO: This requires changes for scalable vectors
/// Basic lowering implementation for one-to-one rewriting from SVE Ops to
/// LLVM Dialect Ops. Convert the type of the result to an LLVM type, pass
/// operands as is, preserve attributes.
template <typename SourceOp, typename TargetOp>
static LogicalResult
matchAndRewriteOneToOne(const ConvertToLLVMPattern &lowering,
                        LLVMTypeConverter &typeConverter, Operation *op,
                        ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) {
  unsigned numResults = op->getNumResults();

  Type packedType;
  if (numResults != 0) {
    packedType = typeConverter.packFunctionResults(op->getResultTypes());
    if (!packedType)
      return failure();
  }

  auto newOp = rewriter.create<TargetOp>(op->getLoc(), packedType, operands,
                                         op->getAttrs());

  // If the operation produced 0 or 1 result, return them immediately.
  if (numResults == 0)
    return rewriter.eraseOp(op), success();
  if (numResults == 1)
    return rewriter.replaceOp(op, newOp.getOperation()->getResult(0)),
           success();

  // Otherwise, it had been converted to an operation producing a structure.
  // Extract individual results from the structure and return them as list.
  SmallVector<Value, 4> results;
  results.reserve(numResults);
  for (unsigned i = 0; i < numResults; ++i) {
    auto type = typeConverter.convertType(op->getResult(i).getType());
    results.push_back(rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), type, newOp.getOperation()->getResult(0),
        rewriter.getI64ArrayAttr(i)));
  }
  rewriter.replaceOp(op, results);
  return success();
}

namespace {
// TODO: Add SVE operations
struct UmmlaOpConversion : public ConvertToLLVMPattern {
  explicit UmmlaOpConversion(MLIRContext *context,
                             LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(UmmlaOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Not the right naming convention for operands
    if (!(getAccVectorElementType(cast<UmmlaOp>(op)).isInteger(32) &&
          getSrc1VectorElementType(cast<UmmlaOp>(op)).isInteger(8) &&
          getSrc2VectorElementType(cast<UmmlaOp>(op)).isInteger(8)))
      return failure();
    return matchAndRewriteOneToOne<UmmlaOp,
                                   LLVM::aarch64_sve_ummla>(
          *this, this->typeConverter, op, operands, rewriter);
  }
};

struct SmmlaOpConversion : public ConvertToLLVMPattern {
  explicit SmmlaOpConversion(MLIRContext *context,
                             LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(SmmlaOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Not the right naming convention for operands
    if (!(getAccVectorElementType(cast<SmmlaOp>(op)).isInteger(32) &&
          getSrc1VectorElementType(cast<SmmlaOp>(op)).isInteger(8) &&
          getSrc2VectorElementType(cast<SmmlaOp>(op)).isInteger(8)))
      return failure();
    return matchAndRewriteOneToOne<SmmlaOp,
                                   LLVM::aarch64_sve_smmla>(
          *this, this->typeConverter, op, operands, rewriter);
  }
};

struct SdotOpConversion : public ConvertToLLVMPattern {
  explicit SdotOpConversion(MLIRContext *context,
                            LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(SdotOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Not the right naming convention for operands
    // TODO: Not the right type checking convention
    if (!((getAccVectorElementType(cast<SdotOp>(op)).isInteger(32) &&
          getSrc1VectorElementType(cast<SdotOp>(op)).isInteger(8) &&
          getSrc2VectorElementType(cast<SdotOp>(op)).isInteger(8)) ||
          (getAccVectorElementType(cast<SdotOp>(op)).isInteger(64) &&
          getSrc1VectorElementType(cast<SdotOp>(op)).isInteger(16) &&
          getSrc2VectorElementType(cast<SdotOp>(op)).isInteger(16))))
      return failure();
    return matchAndRewriteOneToOne<SdotOp,
                                   LLVM::aarch64_sve_sdot>(
          *this, this->typeConverter, op, operands, rewriter);
  }
};

struct UdotOpConversion : public ConvertToLLVMPattern {
  explicit UdotOpConversion(MLIRContext *context,
                            LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(UdotOp::getOperationName(), context,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Not the right naming convention for operands
    // TODO: Not the right type checking convention
    if (!((getAccVectorElementType(cast<UdotOp>(op)).isInteger(32) &&
          getSrc1VectorElementType(cast<UdotOp>(op)).isInteger(8) &&
          getSrc2VectorElementType(cast<UdotOp>(op)).isInteger(8)) ||
          (getAccVectorElementType(cast<UdotOp>(op)).isInteger(64) &&
          getSrc1VectorElementType(cast<UdotOp>(op)).isInteger(16) &&
          getSrc2VectorElementType(cast<UdotOp>(op)).isInteger(16))))
      return failure();
    return matchAndRewriteOneToOne<UdotOp,
                                   LLVM::aarch64_sve_udot>(
          *this, this->typeConverter, op, operands, rewriter);
  }
};

} // namespace

/// Populate the given list with patterns that convert from SVE to LLVM.
void mlir::populateSVEToLLVMConversionPatterns(
    SVETypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  // clang-format off
  patterns.insert<UmmlaOpConversion,
                  SmmlaOpConversion,
                  UdotOpConversion,
                  SdotOpConversion>(ctx, converter);
  // clang-format on
}

namespace {
struct ConvertSVEToLLVMPass
    : public ConvertSVEToLLVMBase<ConvertSVEToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

// Extract an LLVM IR type from the LLVM IR dialect type.
static LLVM::LLVMType unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  auto wrappedLLVMType = type.dyn_cast<LLVM::LLVMType>();
  if (!wrappedLLVMType)
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type");
  return wrappedLLVMType;
}

static Optional<Type> convertScalableVectorType(ScalableVectorType svType,
                                        LLVMTypeConverter &converter)
{
  auto elementType = unwrap(converter.convertType(svType.getElementType()));
  if (!elementType)
    return {};
  // TODO: Force 1D vector
  auto sVectorType = LLVM::LLVMScalableVectorType::get(elementType,
                                                  svType.getShape().back());
/*  auto shape = svType.getShape();
  for (int i = shape.size() - 2; i >= 0; --i)
    sVectorType = LLVM::LLVMType::getArrayTy(sVectorType, shape[i]);*/
  return sVectorType;
}

SVETypeConverter::SVETypeConverter(MLIRContext *ctx)
  : LLVMTypeConverter(ctx) {
    addConversion([this](ScalableVectorType svType) {
      return convertScalableVectorType(svType, *this);
    });
}

void ConvertSVEToLLVMPass::runOnOperation() {
  // Convert to the LLVM IR dialect.
  OwningRewritePatternList patterns;
  SVETypeConverter sveConverter(&getContext());
  populateSVEToLLVMConversionPatterns(sveConverter, patterns);
  populateVectorToLLVMConversionPatterns(sveConverter, patterns);
  populateStdToLLVMConversionPatterns(sveConverter, patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<LLVM::LLVMSVEDialect>();
  target.addIllegalDialect<sve::SVEDialect>();
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertSVEToLLVMPass() {
  return std::make_unique<ConvertSVEToLLVMPass>();
}
