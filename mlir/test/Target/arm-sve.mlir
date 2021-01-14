// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | mlir-translate --arm-sve-mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_sdot
llvm.func @arm_sve_sdot(%arg0: !llvm.vec<?x16 x i8>,
                        %arg1: !llvm.vec<?x16 x i8>,
                        %arg2: !llvm.vec<?x4 x i32>)
                        -> !llvm.vec<?x4 x i32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.sdot.nxv4i32(<vscale x 4
  %0 = "llvm_arm_sve.sdot"(%arg2, %arg0, %arg1) :
    (!llvm.vec<?x4 x i32>, !llvm.vec<?x16 x i8>, !llvm.vec<?x16 x i8>)
        -> !llvm.vec<?x4 x i32>
  llvm.return %0 : !llvm.vec<?x4 x i32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_smmla
llvm.func @arm_sve_smmla(%arg0: !llvm.vec<?x16 x i8>,
                         %arg1: !llvm.vec<?x16 x i8>,
                         %arg2: !llvm.vec<?x4 x i32>)
                         -> !llvm.vec<?x4 x i32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.smmla.nxv4i32(<vscale x 4
  %0 = "llvm_arm_sve.smmla"(%arg2, %arg0, %arg1) :
    (!llvm.vec<?x4 x i32>, !llvm.vec<?x16 x i8>, !llvm.vec<?x16 x i8>)
        -> !llvm.vec<?x4 x i32>
  llvm.return %0 : !llvm.vec<?x4 x i32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_udot
llvm.func @arm_sve_udot(%arg0: !llvm.vec<?x16 x i8>,
                        %arg1: !llvm.vec<?x16 x i8>,
                        %arg2: !llvm.vec<?x4 x i32>)
                        -> !llvm.vec<?x4 x i32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.udot.nxv4i32(<vscale x 4
  %0 = "llvm_arm_sve.udot"(%arg2, %arg0, %arg1) :
    (!llvm.vec<?x4 x i32>, !llvm.vec<?x16 x i8>, !llvm.vec<?x16 x i8>)
        -> !llvm.vec<?x4 x i32>
  llvm.return %0 : !llvm.vec<?x4 x i32>
}

// CHECK-LABEL: define <vscale x 4 x i32> @arm_sve_ummla
llvm.func @arm_sve_ummla(%arg0: !llvm.vec<?x16 x i8>,
                         %arg1: !llvm.vec<?x16 x i8>,
                         %arg2: !llvm.vec<?x4 x i32>)
                         -> !llvm.vec<?x4 x i32> {
  // CHECK: call <vscale x 4 x i32> @llvm.aarch64.sve.ummla.nxv4i32(<vscale x 4
  %0 = "llvm_arm_sve.ummla"(%arg2, %arg0, %arg1) :
    (!llvm.vec<?x4 x i32>, !llvm.vec<?x16 x i8>, !llvm.vec<?x16 x i8>)
        -> !llvm.vec<?x4 x i32>
  llvm.return %0 : !llvm.vec<?x4 x i32>
}

// CHECK-LABEL: define i64 @get_vector_scale()
llvm.func @get_vector_scale() -> i64 {
  // CHECK: call i64 @llvm.vscale.i64()
  %0 = "llvm_arm_sve.vscale"() : () -> i64
  llvm.return %0 : i64
}

// CHECK-LABEL: define <vscale x 8 x i16> @arm_sve_addi
llvm.func @arm_sve_addi(%arg0: !llvm.vec<? x 8 x i16>,
                        %arg1: !llvm.vec<? x 8 x i16>)
                        -> !llvm.vec<? x 8 x i16> {
  // CHECK: add <vscale x 8 x i16>
  %0 = llvm.add %arg0, %arg1  : !llvm.vec<? x 8 x i16>
  llvm.return %0 : !llvm.vec<? x 8 x i16>
}

// CHECK-LABEL: define <vscale x 4 x float> @arm_sve_addf
llvm.func @arm_sve_addf(%arg0: !llvm.vec<? x 4 x f32>,
                        %arg1: !llvm.vec<? x 4 x f32>)
                        -> !llvm.vec<? x 4 x f32> {
  // CHECK: fadd <vscale x 4 x float>
  %0 = llvm.fadd %arg0, %arg1  : !llvm.vec<? x 4 x f32>
  llvm.return %0 : !llvm.vec<? x 4 x f32>
}

// CHECK-LABEL: define <vscale x 8 x i16> @arm_sve_muli
llvm.func @arm_sve_muli(%arg0: !llvm.vec<? x 8 x i16>,
                        %arg1: !llvm.vec<? x 8 x i16>)
                        -> !llvm.vec<? x 8 x i16> {
  // CHECK: mul <vscale x 8 x i16>
  %0 = llvm.mul %arg0, %arg1  : !llvm.vec<? x 8 x i16>
  llvm.return %0 : !llvm.vec<? x 8 x i16>
}

// CHECK-LABEL: define <vscale x 4 x float> @arm_sve_mulf
llvm.func @arm_sve_mulf(%arg0: !llvm.vec<? x 4 x f32>,
                        %arg1: !llvm.vec<? x 4 x f32>)
                        -> !llvm.vec<? x 4 x f32> {
  // CHECK: fmul <vscale x 4 x float>
  %0 = llvm.fmul %arg0, %arg1  : !llvm.vec<? x 4 x f32>
  llvm.return %0 : !llvm.vec<? x 4 x f32>
}
