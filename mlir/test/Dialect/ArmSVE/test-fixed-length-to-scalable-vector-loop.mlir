// RUN: mlir-opt %s --arm-sve-fixed-length-to-vla-loop | mlir-opt | FileCheck %s 

func @vector_add(%a : memref<2048xf32>,
                 %b : memref<2048xf32>,
                 %c : memref<2048xf32> ) {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  // CHECK: arm_sve.vector_scale : index
  // CHECK: [[SCALABLESTEP:%[0-9a-z]+]] = muli {{.*}}: index
  %c2048 = constant 2048 : index
  // CHECK: scf.for {{.*}} step [[SCALABLESTEP]]
  scf.for %i0 = %c0 to %c2048 step %c4 {
    // CHECK: arm_sve.load {{.*}}: !arm_sve.vector<4xf32> from memref<2048xf32>
    %0 = vector.load %a[%i0] : memref<2048xf32>, vector<4xf32>
    // CHECK: arm_sve.load {{.*}}: !arm_sve.vector<4xf32> from memref<2048xf32>
    %1 = vector.load %b[%i0] : memref<2048xf32>, vector<4xf32>
    // CHECK: arm_sve.addf {{.*}}: !arm_sve.vector<4xf32>
    %2 = addf %0, %1 : vector<4xf32>
    // CHECK: arm_sve.store {{.*}}: !arm_sve.vector<4xf32> to memref<2048xf32>
    vector.store %2, %c[%i0] : memref<2048xf32>, vector<4xf32>
  } { "vector.promote_to_vla" = true }
  return
}
