//===-- ConvertContractToSVEMMLA.cpp - conversion of loops to VLA loops ---===//
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"

using namespace mlir;

namespace {

#define DEBUG_TYPE "ArmSVE"

bool vectorDefinedByExtOpFromType(Value val, Type elTy) {
  if (auto defOp = dyn_cast<arith::ExtSIOp>(val.getDefiningOp()))
    return defOp.getIn().getType().cast<VectorType>().getElementType() == elTy;
  else if (auto defOp = dyn_cast<arith::ExtUIOp>(val.getDefiningOp()))
    return defOp.getIn().getType().cast<VectorType>().getElementType() == elTy;
  else if (auto defOp = dyn_cast<arith::ExtFOp>(val.getDefiningOp()))
    return defOp.getIn().getType().cast<VectorType>().getElementType() == elTy;
  return false;
}

// Check if the type of the value is a vector of the right shape and type, or is
// defined by a cast operation from a vector of the right shape and type.
bool vectorTypeMatchesShapeAndType(Value val, ArrayRef<int64_t> dim,
                                   Type elTy) {
  auto vType = val.getType().cast<VectorType>();
  if (vType.getShape() != dim) return false;
  if (vType.getElementType() == elTy) return true;
  return vectorDefinedByExtOpFromType(val, elTy);
}

// Do operands match SMMLA/UMMLA/USMMLA?
bool typesMatchIntMMLA(Value lhs, Value rhs, Value acc) {
  auto i8ty = IntegerType::get(lhs.getContext(), 8);
  auto i32ty = IntegerType::get(lhs.getContext(), 32);
  return vectorTypeMatchesShapeAndType(lhs, ArrayRef<int64_t>{2,8}, i8ty) &&
         vectorTypeMatchesShapeAndType(rhs, ArrayRef<int64_t>{8,2}, i8ty) &&
         vectorTypeMatchesShapeAndType(acc, ArrayRef<int64_t>{2,2}, i32ty);
}

// Do operands match BFMMLA?
bool typesMatchBFloatMMLA(Value lhs, Value rhs, Value acc) {
  auto bf16ty = FloatType::getBF16(lhs.getContext());
  auto f32ty = FloatType::getF32(lhs.getContext());
  return vectorTypeMatchesShapeAndType(lhs, ArrayRef<int64_t>{2,4}, bf16ty) &&
         vectorTypeMatchesShapeAndType(rhs, ArrayRef<int64_t>{4,2}, bf16ty) &&
         vectorTypeMatchesShapeAndType(acc, ArrayRef<int64_t>{2,2}, f32ty);
}

// Do operands match FMMLA?
bool typesMatchFloatMMLA(Value lhs, Value rhs, Value acc) {
  auto f32ty = FloatType::getF32(lhs.getContext());
  return vectorTypeMatchesShapeAndType(lhs, ArrayRef<int64_t>{2,2}, f32ty) &&
         vectorTypeMatchesShapeAndType(rhs, ArrayRef<int64_t>{2,2}, f32ty) &&
         vectorTypeMatchesShapeAndType(acc, ArrayRef<int64_t>{2,2}, f32ty);
}

// TODO: Just compare the types, ffs!!!
//       <- can't! It's type or type cast from
// Check if the operands match one of the available xMMLA intrinsics
bool typesMatchMMLA(Value lhs, Value rhs, Value acc) {
  return typesMatchIntMMLA(lhs, rhs, acc) ||
         typesMatchBFloatMMLA(lhs, rhs, acc) ||
         typesMatchFloatMMLA(lhs, rhs, acc);
}

// Check if the vector contraction is a Matrix-Matrix Multiply and Accumulate op
bool isMMMATypeContractionOp(vector::ContractionOp& op) {
  auto op_it_types = op.getIteratorTypes().getValue();
  auto gemm_it_types = ArrayRef<Attribute>({
    StringAttr::get(op.getContext(), "parallel"),
    StringAttr::get(op.getContext(), "parallel"),
    StringAttr::get(op.getContext(), "reduction")});
  if (op_it_types != gemm_it_types)
    return false;
  // Check it's an `<add>` reduction
  if (op.getKind() != vector::ContractionOp::getDefaultKind())
    return false;
  // Check indexing_maps are:
  //     (d0, d1, d2) -> (d0, d2)
  //     (d0, d1, d2) -> (d2, d1)
  //     (d0, d1, d2) -> (d0, d1)
  std::vector<DenseMap<int64_t, int64_t>> indexMap;
  op.getIterationIndexMap(indexMap);
  if (indexMap[0][0] != 0 || indexMap[0][2] != 1 ||
      indexMap[1][1] != 1 || indexMap[1][2] != 0 ||
      indexMap[2][0] != 0 || indexMap[2][1] != 1)
      return false;
  return true;
}

bool hasMMLASemantics(vector::ContractionOp& op) {
  // Verify it's the right kind of vector contraction
  if (!isMMMATypeContractionOp(op))
    return false;
  // Verify it's one of the available intrinsics
  // Check operands are 2x8, 8x2, 2x2
  //      TODO: operands can be 2x4, 4x2, 2x2 for 16bit bf16 contractions
  //            operands can be 2x2, 2x2, 2x2 for 32bit f32 contractions
  return typesMatchMMLA(op.getLhs(), op.getRhs(), op.getAcc());
}

Value getUnCastedOperandValue(Value val, Type elTy) {
  if (val.getType().cast<VectorType>().getElementType() == elTy)
    return val;
  if (auto defOp = dyn_cast<arith::ExtSIOp>(val.getDefiningOp()))
    return defOp.getIn();
  else if (auto defOp = dyn_cast<arith::ExtUIOp>(val.getDefiningOp()))
    return defOp.getIn();
  return cast<arith::ExtFOp>(val.getDefiningOp()).getIn();
}

struct ConvertVectorContractionToSVEMMLA :
                public OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern<vector::ContractionOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto acc = adaptor.getAcc();
    // For now, we allow only SMMLA
    if (!typesMatchIntMMLA(lhs, rhs, acc))
      return failure();
    auto i8ty = rewriter.getI8Type();
    auto i32ty = rewriter.getI32Type();
    auto fInVTy = VectorType::get({16}, i8ty, 0);
    auto fAccVTy = VectorType::get({4}, i32ty, 0);
    auto sInVTy = VectorType::get({16}, i8ty, 1);
    auto sOutVTy = VectorType::get({4}, i32ty, 1);
    auto outVTy2d = VectorType::get({2, 2}, i32ty);
    lhs = getUnCastedOperandValue(adaptor.getLhs(), i8ty);
    rhs = getUnCastedOperandValue(adaptor.getRhs(), i8ty);
    acc = getUnCastedOperandValue(adaptor.getAcc(), i32ty);
    auto loc = contractOp.getLoc();
    // Create constant vectors to insert fixed-length to scalable
    auto sCst16v = rewriter.create<arith::ConstantOp>(loc, sInVTy,
                              rewriter.getZeroAttr(sInVTy));
    auto sCst4v = rewriter.create<arith::ConstantOp>(loc, sOutVTy,
                              rewriter.getZeroAttr(sOutVTy));
    // Convert lhs from a 2D fixed-length into a 1D scalable vector
    auto lhsFlat = rewriter.create<vector::ShapeCastOp>(loc, fInVTy, lhs);
    auto lhsScal = rewriter.create<vector::ScalableInsertOp>(loc, lhsFlat, sCst16v, 0);
    // Convert lhs from a 2D fixed-length into a 1D scalable vector
    auto rhsFlat = rewriter.create<vector::ShapeCastOp>(loc, fInVTy, rhs);
    auto rhsScal = rewriter.create<vector::ScalableInsertOp>(loc, rhsFlat, sCst16v, 0);
    // Convert acc from a 2D fixed-length into a 1D scalable vector
    auto accFlat = rewriter.create<vector::ShapeCastOp>(loc, fAccVTy, acc);
    auto accScal = rewriter.create<vector::ScalableInsertOp>(loc, accFlat, sCst4v, 0);
    // Insert `smmla`to replace `vector.contract`
    auto mmlaop = rewriter.create<arm_sve::SmmlaOp>(loc, sOutVTy, accScal, lhsScal, rhsScal);
    // Convert result from a 1D scalable vector into a 2D fixed-length vector
    auto outFlat = rewriter.create<vector::ScalableExtractOp>(loc, fAccVTy, mmlaop, 0);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(contractOp, outVTy2d, outFlat);
    return success();
  }
};

class ConvertVectorContractionToSVEMMLAPass
    : public ConvertVectorContractionToSVEMMLABase<
              ConvertVectorContractionToSVEMMLAPass> {
  using ConvertVectorContractionToSVEMMLABase<
  ConvertVectorContractionToSVEMMLAPass>::ConvertVectorContractionToSVEMMLABase;

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addDynamicallyLegalOp<vector::ContractionOp>(
      [](vector::ContractionOp op) {
        return !hasMMLASemantics(op);
      });
    target.addLegalDialect<arm_sve::ArmSVEDialect, func::FuncDialect,
      arith::ArithmeticDialect, scf::SCFDialect, memref::MemRefDialect,
      vector::VectorDialect>();
    patterns.add<ConvertVectorContractionToSVEMMLA>(patterns.getContext());
    if (failed(applyFullConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertVectorContractionToSVEMMLAPass() {
  return std::make_unique<ConvertVectorContractionToSVEMMLAPass>();
}