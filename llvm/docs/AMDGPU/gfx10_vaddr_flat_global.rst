..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid10_vaddr_flat_global:

vaddr
===========================

A 64-bit flat global address or a 32-bit offset depending on addressing mode:

* Address = :ref:`vaddr<amdgpu_synid10_vaddr_flat_global>` + :ref:`offset12s<amdgpu_synid_flat_offset12s>`. :ref:`vaddr<amdgpu_synid10_vaddr_flat_global>` is a 64-bit address. This mode is indicated by :ref:`saddr<amdgpu_synid10_saddr_flat_global>` set to :ref:`off<amdgpu_synid_off>`.
* Address = :ref:`saddr<amdgpu_synid10_saddr_flat_global>` + :ref:`vaddr<amdgpu_synid10_vaddr_flat_global>` + :ref:`offset12s<amdgpu_synid_flat_offset12s>`. :ref:`vaddr<amdgpu_synid10_vaddr_flat_global>` is a 32-bit offset. This mode is used when :ref:`saddr<amdgpu_synid10_saddr_flat_global>` is not :ref:`off<amdgpu_synid_off>`.

*Size:* 1 or 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`
