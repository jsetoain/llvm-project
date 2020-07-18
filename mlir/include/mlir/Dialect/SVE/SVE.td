//===-- SVE.td - SVE dialect operation definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the SVE dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SVE_OPS
#define SVE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SVE dialect definition
//===----------------------------------------------------------------------===//

def SVE_Dialect : Dialect {
  let name = "sve";
  let cppNamespace = "::mlir::sve";
  let summary = "Basic dialect to target Arm SVE architectures";
  let description = [{
    This dialect contains the definitions necessary to target SVE scalable
    vector operations, including a scalable vector type and intrinsics for
    some SVE instructions.
  }];
}

//===----------------------------------------------------------------------===//
// SVE scalable vector type definition
//===----------------------------------------------------------------------===//

def SVE_ScalableVectorType : DialectType<SVE_Dialect,
    CPred<"$_self.isa<ScalableVectorType>()">,
              "scalable vector type">,
    BuildableType<"$_builder.getType<ScalableVectorType>()"> {
  let typeDescription = [{
    `sve.vector` represents vectors that will be processed by an scalable
    vector architecture.
  }];
}

def IsScalableVectorTypePred :
    CPred<"$_self.isa<::mlir::sve::ScalableVectorType>()">;

class ScalableVectorOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>, IsScalableVectorTypePred,
              "$_self.cast<::mlir::sve::ScalableVectorType>().getElementType()",
              "scalable vector">;

//===----------------------------------------------------------------------===//
// SVE op definitions
//===----------------------------------------------------------------------===//

class SVE_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<SVE_Dialect, mnemonic, traits> {}

def UmmlaOp : SVE_Op<"ummla",
                [NoSideEffect,
                AllTypesMatch<["src1", "src2"]>,
                AllTypesMatch<["acc", "dst"]>,
              ]> {
  let summary = "Matrix-matrix mutiply and accumulate op";
  let description = [{
    The ummla op is an SVE specific op that can lower to the proper LLVMSVE
    operation: `llvm.aarch64.sve.ummla` instruction.
    TODO: Add documentation for ummla.
  }];
  // Supports vector<16xi8>.
  let arguments = (ins
          ScalableVectorOf<[UI32]>:$acc,
          ScalableVectorOf<[UI8]>:$src1,
          ScalableVectorOf<[UI8]>:$src2
  );
  let results = (outs ScalableVectorOf<[UI32]>:$dst);
  let assemblyFormat =
    "$acc `,` $src1 `,` $src2 attr-dict `:` type($src1) `->` type($dst)";
}

def SmmlaOp : SVE_Op<"smmla",
                [NoSideEffect,
                AllTypesMatch<["src1", "src2"]>,
                AllTypesMatch<["acc", "dst"]>,
              ]> {
  let summary = "Matrix-matrix mutiply and accumulate op";
  let description = [{
    The ummla op is an SVE specific op that can lower to the proper LLVMSVE
    operation: `llvm.aarch64.sve.smmla` instruction.
    TODO: Add documentation for smmla.
  }];
  // Supports vector<16 x i8>.
  let arguments = (ins
          ScalableVectorOf<[SI32]>:$acc,
          ScalableVectorOf<[SI8]>:$src1,
          ScalableVectorOf<[SI8]>:$src2
  );
  let results = (outs ScalableVectorOf<[SI32]>:$dst);
  let assemblyFormat =
    "$acc `,` $src1 `,` $src2 attr-dict `:` type($src1) `->` type($dst)";
}

def SdotOp : SVE_Op<"sdot",
               [NoSideEffect,
               AllTypesMatch<["src1", "src2"]>,
               AllTypesMatch<["acc", "dst"]>,
             ]> {
  let summary = "Vector-vector dot product and accumulate op";
  let description = [{
    The ummla op is an SVE specific op that can lower to the proper LLVMSVE
    operation: `llvm.aarch64.sve.sdot`
    TODO: Add documentation for sdot.
  }];
  // Supports vector<16xi8> and vector<8xi16>.
  let arguments = (ins
          ScalableVectorOf<[SI32, SI64]>:$acc,
          ScalableVectorOf<[SI8, SI16]>:$src1,
          ScalableVectorOf<[SI8, SI16]>:$src2
  );
  let results = (outs ScalableVectorOf<[SI32, SI64]>:$dst);
  let assemblyFormat =
    "$acc `,` $src1 `,` $src2 attr-dict `:` type($src1) `->` type($dst)";
}

def UdotOp : SVE_Op<"udot",
               [NoSideEffect,
               AllTypesMatch<["src1", "src2"]>,
               AllTypesMatch<["acc", "dst"]>,
             ]> {
  let summary = "Vector-vector dot product and accumulate op";
  let description = [{
    The ummla op is an SVE specific op that can lower to the proper LLVMSVE
    operation: `llvm.aarch64.sve.udot`
    TODO: Add documentation for sdot.
  }];
  // Supports vector<16xi8> and vector<8xi16>.
  let arguments = (ins
          ScalableVectorOf<[UI32, UI64]>:$acc,
          ScalableVectorOf<[UI8, UI16]>:$src1,
          ScalableVectorOf<[UI8, UI16]>:$src2
  );
  let results = (outs ScalableVectorOf<[UI32, UI64]>:$dst);
  let assemblyFormat =
    "$acc `,` $src1 `,` $src2 attr-dict `:` type($src1) `->` type($dst)";
}

#endif // SVE_OPS
