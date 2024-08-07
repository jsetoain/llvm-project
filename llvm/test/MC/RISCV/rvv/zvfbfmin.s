# RUN: llvm-mc -triple=riscv32 -show-encoding -mattr=+zvfbfmin %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+zvfbfmin %s \
# RUN:    | llvm-objdump -d --mattr=+zvfbfmin - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj -mattr=+zvfbfmin %s \
# RUN:    | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+zvfbfmin %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v,+f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+zvfbfmin %s \
# RUN:    | llvm-objdump -d --mattr=+zvfbfmin - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+zvfbfmin %s \
# RUN:    | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# CHECK-INST: vfncvtbf16.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4e,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvfbfmin' (Vector BF16 Converts){{$}}
# CHECK-UNKNOWN: 484e9457 <unknown>
vfncvtbf16.f.f.w v8, v4, v0.t

# CHECK-INST: vfncvtbf16.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4e,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvfbfmin' (Vector BF16 Converts){{$}}
# CHECK-UNKNOWN: 4a4e9457 <unknown>
vfncvtbf16.f.f.w v8, v4

# CHECK-INST: vfwcvtbf16.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x46,0x48]
# CHECK-ERROR: instruction requires the following: 'Zvfbfmin' (Vector BF16 Converts){{$}}
# CHECK-UNKNOWN: 48469457 <unknown>
vfwcvtbf16.f.f.v v8, v4, v0.t

# CHECK-INST: vfwcvtbf16.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x46,0x4a]
# CHECK-ERROR: instruction requires the following: 'Zvfbfmin' (Vector BF16 Converts){{$}}
# CHECK-UNKNOWN: 4a469457 <unknown>
vfwcvtbf16.f.f.v v8, v4
