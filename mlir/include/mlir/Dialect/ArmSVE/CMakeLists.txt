add_mlir_dialect(ArmSVE arm_sve ArmSVE)
add_mlir_doc(ArmSVE ArmSVE Dialects/ -gen-dialect-doc -dialect=arm_sve)

set(LLVM_TARGET_DEFINITIONS ArmSVE.td)
mlir_tablegen(ArmSVEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRArmSVEConversionsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ArmSVE)
add_public_tablegen_target(MLIRArmSVEPassIncGen)

add_mlir_doc(Passes ArmSVEPasses ./ -gen-pass-doc)