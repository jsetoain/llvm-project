// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

func.func @arm_sve_sdot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>) -> vector<[4]xi32> {
  // CHECK: arm_sve.sdot {{.*}}: vector<[16]xi8> to vector<[4]xi32
  %0 = arm_sve.sdot %c, %a, %b :
             vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @arm_sve_smmla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>) -> vector<[4]xi32> {
  // CHECK: arm_sve.smmla {{.*}}: vector<[16]xi8> to vector<[4]xi3
  %0 = arm_sve.smmla %c, %a, %b :
             vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @arm_sve_udot(%a: vector<[16]xi8>,
                   %b: vector<[16]xi8>,
                   %c: vector<[4]xi32>) -> vector<[4]xi32> {
  // CHECK: arm_sve.udot {{.*}}: vector<[16]xi8> to vector<[4]xi32
  %0 = arm_sve.udot %c, %a, %b :
             vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @arm_sve_ummla(%a: vector<[16]xi8>,
                    %b: vector<[16]xi8>,
                    %c: vector<[4]xi32>) -> vector<[4]xi32> {
  // CHECK: arm_sve.ummla {{.*}}: vector<[16]xi8> to vector<[4]xi3
  %0 = arm_sve.ummla %c, %a, %b :
             vector<[16]xi8> to vector<[4]xi32>
  return %0 : vector<[4]xi32>
}
