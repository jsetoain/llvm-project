// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sve" -convert-func-to-llvm -reconcile-unrealized-casts | mlir-opt | FileCheck %s

func.func @arm_sve_sdot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.sdot
  %0 = arm_sve.sdot %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @arm_sve_smmla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.smmla
  %0 = arm_sve.smmla %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @arm_sve_udot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.udot
  %0 = arm_sve.udot %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @arm_sve_ummla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>)
    -> vector<[4]xi32> {
  // CHECK: arm_sve.intr.ummla
  %0 = arm_sve.ummla %c, %a, %b :
               vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @get_vector_scale() -> index {
  // CHECK: llvm.intr.vscale
  %0 = vector.vscale
  return %0 : index
}
