add_mlir_dialect_library(MLIRArmSVETransforms
  ConvertFixedLengthToVLALoop.cpp
  LegalizeForLLVMExport.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSVE

  DEPENDS
  MLIRArmSVEConversionsIncGen
  MLIRArmSVEPassIncGen

  LINK_LIBS PUBLIC
  MLIRArmSVE
  MLIRIR
  MLIRMemRef
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRTransformUtils
  MLIRVector
  )
