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

def ConvertVectorContractionToSVEMMLA :
                Pass<"arm-sve-vls-vector-contract-to-mmla", "func::FuncOp"> {
  let summary = "Convert VLS (vector-length specific) `vector.contract` to ArmSVE xMMLA intrinsics";
  let description = [{
    Convert all `vector.contract` ops with the right operands into ArmSVE xMMLA
    intrinsics in a VLS context.

    A vector contraction with the following contraction trait:
    ```mlir
    #mmlatrait = {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = [
        "parallel",
        "parallel",
        "reduction"],
      kind = #vector.kind<add>
    }```
    And one of the following signatures:
    ```mlir
    %0 = vector.contract %lhs, %rhs, %acc #mmlatrait: vector<2x8xi8>, vector<8x2xi8> into vector<2x2xi32>
    %1 = vector.contract %lhs, %rhs, %acc #mmlatrait: vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
    %2 = vector.contract %lhs, %rhs, %acc #mmlatrait: vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
    ```
    Has the same semantics as one of the MMLA Arm SVE intrinsic variants.

    This pass locates such occurences and converts them to the appropriate
    arm_sve.<x>mmla operation. This pass will also fold trivial operator
    widenings into the `vector.contract` op before converting. E.g.:
    ```mlir
     %0 = arith.extsi %lhs : vector<2x8xi8> to vector<2x8xi32>
     %1 = arith.extsi %rhs : vector<8x2xi8> to vector<8x2xi32>
     %res = vector.contract %0, %1, %acc #mmlatrait: vector<2x8xi32>, vector<8x2xi32> into vector<2x2xi32>```
     Will be folded into:
     ```mlir
     %res = vector.contract %lhs, %rhs, %acc #mmlatrait: vector<2x8xi8>, vector<8x2xi8> into vector<2x2xi32>```
     ```
     And then converted into:
     ```mlir
     %cst4 = arith.constant dense<0> : vector<[4]xi32>
     %cst16 = arith.constant dense<0> : vector<[16]xi8>
     %flhs = vector.shape_cast %lhs : vector<2x8xi8> to vector<16xi8>
     %frhs = vector.shape_cast %rhs : vector<2x8xi8> to vector<16xi8>
     %facc = vector.shape_cast %acc : vector<2x2xi32> to vector<4xi32>
     %slhs = vector.scalable.insert %flhs, %cst16[0] : vector<16xi8> into vector<[16]xi8>
     %srhs = vector.scalable.insert %frhs, %cst16[0] : vector<16xi8> into vector<[16]xi8>
     %sacc = vector.scalable.insert %facc, %cst4[0] : vector<4xi32> into vector<[4]xi32>
     %sres = arm_sve.smmla %sacc, %slhs, %srhs : vector<[16]xi8> to vector<[4]xi32>
     %fres = vector.scalable.extract %sres[0] : vector<4xi32> from vector<[4]xi32>
     %res = vector.shape_cast %resf : vector<4xi32> to vector<2x2xi32>
     ```
  }];
  let constructor = "mlir::createConvertVectorContractionToSVEMMLAPass()";
  let dependentDialects = ["arm_sve::ArmSVEDialect"];
}

#endif // MLIR_DIALECT_ARMSVE_PASSES
