//===- ConvertFixedLengthToVLALoop.cpp - conversion of loops to VLA loops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Passes.h"
#include "mlir/Dialect/ArmSVE/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"

using namespace mlir;

const char *mlir::arm_sve::kPromoteLoopToVLALoop = "vector.promote_to_vla";

namespace {

#define DEBUG_TYPE "ArmSVE"

class VLATypeConverter : public TypeConverter {
public:
  VLATypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](VectorType vType) -> arm_sve::ScalableVectorType {
      // TODO: Add type conversion validity check
      //           * VectorType has an ElementType
      //           * VectorType is rank 1
      //           * VectorType Shape is compatible with minimum vector size (e.g.: 128-bit)
      arm_sve::ScalableVectorType svType = arm_sve::ScalableVectorType::get(
                          vType.getContext(),
                          vType.getShape().back(),
                          vType.getElementType());
      return svType;
    });
  }
};

bool dependsOnVscale(Value val) {
  Operation* op = val.getDefiningOp();
  if (op) {
    if (isa<arm_sve::VectorScaleOp>(op))
      return true;
    for (Value operand : op->getOperands()) {
      if (dependsOnVscale(operand))
        return true;
    }
  }
  return false;
}

bool isLoopMarkedVLA(scf::ForOp loop) {
  auto prom2vla = loop->getAttrOfType<BoolAttr>(arm_sve::kPromoteLoopToVLALoop);
  if (prom2vla && prom2vla.getValue())
    return true;
  return false;
}

bool isOpInVLALoop(Operation *op) {
  Operation* parent = op->getParentOp();
  while (parent) {
    if (isa<scf::ForOp>(parent)) {
      scf::ForOp loop = dyn_cast<scf::ForOp>(parent);
      if (isLoopMarkedVLA(loop) || dependsOnVscale(loop.step())) {
        return true;
      }
    }
    parent = parent->getParentOp();
  }
  return false;
}

template <typename SourceOp, typename TargetOp>
struct StdBinVecOpToScalableBinVecOp :
        public OpConversionPattern<SourceOp> {
    using OpConversionPattern<SourceOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter) const override {
      typename SourceOp::Adaptor adaptor(operands, op->getAttrDictionary());
      rewriter.replaceOpWithNewOp<TargetOp>(op,
                      this->getTypeConverter()->convertType(op.getType()),
                      adaptor.lhs(), adaptor.rhs());
      return success();
    }
};

struct FixedLengthToScalableLoad :
        public OpConversionPattern<vector::LoadOp> {
    using OpConversionPattern<vector::LoadOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(vector::LoadOp op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter) const override {
      vector::LoadOpAdaptor adaptor(operands, op->getAttrDictionary());
      rewriter.replaceOpWithNewOp<arm_sve::ScalableLoadOp>(op,
                      this->getTypeConverter()->convertType(op.getVectorType()),
                      adaptor.base(), adaptor.indices()[0]);
      return success();
    }
};

struct FixedLengthToScalableStore :
        public OpConversionPattern<vector::StoreOp> {
    using OpConversionPattern<vector::StoreOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(vector::StoreOp op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter) const override {
      vector::StoreOpAdaptor adaptor(operands, op->getAttrDictionary());
      rewriter.replaceOpWithNewOp<arm_sve::ScalableStoreOp>(op,
                                adaptor.base(), adaptor.indices()[0],
                                adaptor.valueToStore());
      return success();
    }
};


struct FixedLengthToScalableBroadcast :
        public OpConversionPattern<vector::BroadcastOp> {
    using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(vector::BroadcastOp op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter) const override {
      vector::BroadcastOpAdaptor adaptor(operands, op->getAttrDictionary());
      rewriter.replaceOpWithNewOp<arm_sve::ScalableBroadcastOp>(op,
                      this->getTypeConverter()->convertType(op.getVectorType()),
                      adaptor.source());
      return success();
    }
};

struct MakeLoopVLA : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(scf::ForOp loop, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isLoopMarkedVLA(loop))
      return failure();
    rewriter.startRootUpdate(loop);
    // Probably don't need these
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(loop.step());
    auto loc = loop.step().getLoc();
    Value vscale = rewriter.create<arm_sve::VectorScaleOp>(
      loc, IndexType::get(rewriter.getContext()));
    Value newStep = rewriter.create<arith::MulIOp>(
      loc, vscale, loop.step());
    loop.setStep(newStep);
    loop->removeAttr(arm_sve::kPromoteLoopToVLALoop);
    rewriter.finalizeRootUpdate(loop);
    return success();
  }
};

class ConvertFixedLengthToVLALoopPass
    : public ConvertFixedLengthToVLALoopBase<ConvertFixedLengthToVLALoopPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    VLATypeConverter typeConverter;
    RewritePatternSet patterns(context);
    arm_sve::populateArmSVEVLAConversionPatterns(typeConverter, patterns);
    ConversionTarget target(*context);
    target.addDynamicallyLegalOp<scf::ForOp>(
      [](scf::ForOp op) {
        return !isLoopMarkedVLA(op);
      });
    target.addDynamicallyLegalOp<vector::BroadcastOp>(
      [](vector::BroadcastOp op) {
        return !isOpInVLALoop(op);
      });
    target.addDynamicallyLegalOp<vector::LoadOp>(
      [&](vector::LoadOp op) {
        return !isOpInVLALoop(op);
      });
    target.addDynamicallyLegalOp<vector::StoreOp>(
      [&](vector::StoreOp op) {
        return !isOpInVLALoop(op);
      });
    target.addDynamicallyLegalOp<arith::AddIOp>(
      [&](arith::AddIOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addDynamicallyLegalOp<arith::AddFOp>(
      [&](arith::AddFOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addDynamicallyLegalOp<arith::SubIOp>(
      [&](arith::SubIOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addDynamicallyLegalOp<arith::SubFOp>(
      [&](arith::SubFOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addDynamicallyLegalOp<arith::MulIOp>(
      [&](arith::MulIOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addDynamicallyLegalOp<arith::MulFOp>(
      [&](arith::MulFOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addDynamicallyLegalOp<arith::DivSIOp>(
      [&](arith::DivSIOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addDynamicallyLegalOp<arith::DivUIOp>(
      [&](arith::DivUIOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addDynamicallyLegalOp<arith::DivFOp>(
      [&](arith::DivFOp op) {
        return !(op.getType().isa<VectorType>() && isOpInVLALoop(op));
      });
    target.addLegalDialect<arm_sve::ArmSVEDialect>();
    target.addLegalDialect<scf::SCFDialect>(); // Should do DynamicallyLegal
    target.addLegalDialect<vector::VectorDialect>(); // Should do DynamicallyLegal
    target.addLegalDialect<StandardOpsDialect>(); // Should do DynamicallyLegal
    target.addLegalDialect<BuiltinDialect>();
    if (failed(applyFullConversion(getFunction(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertFixedLengthToVLALoopPass() {
  return std::make_unique<ConvertFixedLengthToVLALoopPass>();
}

void mlir::arm_sve::populateArmSVEVLAConversionPatterns(
                   TypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<MakeLoopVLA,
               StdBinVecOpToScalableBinVecOp<arith::AddIOp, arm_sve::ScalableAddIOp>,
               StdBinVecOpToScalableBinVecOp<arith::AddFOp, arm_sve::ScalableAddFOp>,
               StdBinVecOpToScalableBinVecOp<arith::SubIOp, arm_sve::ScalableSubIOp>,
               StdBinVecOpToScalableBinVecOp<arith::SubFOp, arm_sve::ScalableSubFOp>,
               StdBinVecOpToScalableBinVecOp<arith::MulIOp, arm_sve::ScalableMulIOp>,
               StdBinVecOpToScalableBinVecOp<arith::MulFOp, arm_sve::ScalableMulFOp>,
               StdBinVecOpToScalableBinVecOp<arith::DivSIOp, arm_sve::ScalableSDivIOp>,
               StdBinVecOpToScalableBinVecOp<arith::DivUIOp, arm_sve::ScalableUDivIOp>,
               StdBinVecOpToScalableBinVecOp<arith::DivFOp, arm_sve::ScalableDivFOp>,
               FixedLengthToScalableLoad,
               FixedLengthToScalableStore,
               FixedLengthToScalableBroadcast
              >(typeConverter, patterns.getContext(), 10);
}