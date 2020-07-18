//===-- LLVMSVE.td - LLVMSVE dialect op definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the LLVMSVE dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_SVE_OPS
#define LLVMIR_SVE_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// LLVMSVE dialect definition
//===----------------------------------------------------------------------===//

def LLVMSVE_Dialect : Dialect {
  let name = "llvm_sve";
  let cppNamespace = "::mlir::LLVM";
}

//----------------------------------------------------------------------------//
// MLIR LLVM SVE intrinsics using the MLIR LLVM Dialect type system
//----------------------------------------------------------------------------//

class LLVMSVE_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase<LLVMSVE_Dialect, mnemonic,
                  "aarch64_sve_" # !subst(".", "_", mnemonic),
                  [], [], traits, 1>;

//def LLVM_sve_fadd :
//  LLVMSVE_InstrOp<"fadd">,
//  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_aarch64_sve_ummla :
  LLVMSVE_IntrOp<"ummla">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_aarch64_sve_smmla :
  LLVMSVE_IntrOp<"smmla">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_aarch64_sve_sdot :
  LLVMSVE_IntrOp<"sdot">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_aarch64_sve_udot :
  LLVMSVE_IntrOp<"udot">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

#endif // SVE_OPS
