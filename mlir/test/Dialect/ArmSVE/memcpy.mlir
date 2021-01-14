// RUN: mlir-opt -convert-scf-to-std -convert-vector-to-llvm="enable-arm-sve" -convert-std-to-llvm % | mlir-translate -arm-sve-mlir-to-llvmir | llc -march=aarch64 -mattr=sve
func @memcopy(%src : memref<?xf32>, %dst : memref<?xf32>, %size : index) {
  %vs = arm_sve.vector_scale : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %step = muli %c4, %vs : index

  scf.for %i0 = %c0 to %size step %step {
    %0 = arm_sve.load %src[%i0] : memref<?xf32> into !arm_sve.vector<4xf32>
    arm_sve.store %dst[%i0], %0 : memref<?xf32>, !arm_sve.vector<4xf32>
  }

  return
}
