"""Tests for ortpy.MemoryInfo."""
import pytest

import ortpy as ort


class TestMemoryInfoDefaultCpu:
    def test_default_construction(self):
        mem = ort.MemoryInfo()
        assert mem is not None

    def test_default_name(self):
        mem = ort.MemoryInfo()
        assert mem.name == "Cpu"

    def test_default_device_id(self):
        mem = ort.MemoryInfo()
        assert mem.device_id == 0

    def test_default_mem_type(self):
        mem = ort.MemoryInfo()
        assert mem.mem_type == ort.MemType.DEFAULT

    def test_default_allocator_type(self):
        mem = ort.MemoryInfo()
        # Default CPU allocator uses arena
        assert mem.allocator_type == ort.AllocatorType.ARENA

    def test_default_device_type(self):
        mem = ort.MemoryInfo()
        assert mem.device_type == ort.MemoryInfoDeviceType.CPU


class TestMemoryInfoCustom:
    def test_custom_construction(self):
        mem = ort.MemoryInfo("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        assert mem.name == "Cpu"
        assert mem.allocator_type == ort.AllocatorType.ARENA
        assert mem.device_id == 0
        assert mem.mem_type == ort.MemType.DEFAULT

    def test_cpu_input_mem_type(self):
        mem = ort.MemoryInfo("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.CPU_INPUT)
        assert mem.mem_type == ort.MemType.CPU_INPUT

    def test_cpu_output_mem_type(self):
        mem = ort.MemoryInfo("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.CPU_OUTPUT)
        assert mem.mem_type == ort.MemType.CPU_OUTPUT


class TestMemoryInfoComparison:
    def test_equal_default(self):
        a = ort.MemoryInfo()
        b = ort.MemoryInfo()
        assert a == b

    def test_equal_custom_same(self):
        a = ort.MemoryInfo("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        b = ort.MemoryInfo("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        assert a == b

    def test_not_equal_different_allocator(self):
        a = ort.MemoryInfo("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.DEFAULT)
        b = ort.MemoryInfo("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        assert not (a == b)

    def test_not_equal_different_mem_type(self):
        a = ort.MemoryInfo("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.DEFAULT)
        b = ort.MemoryInfo("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.CPU_INPUT)
        assert not (a == b)


class TestMemoryInfoEnums:
    def test_allocator_type_values(self):
        assert ort.AllocatorType.INVALID is not None
        assert ort.AllocatorType.DEVICE is not None
        assert ort.AllocatorType.ARENA is not None

    def test_mem_type_values(self):
        assert ort.MemType.CPU_INPUT is not None
        assert ort.MemType.CPU_OUTPUT is not None
        assert ort.MemType.DEFAULT is not None

    def test_device_type_values(self):
        assert ort.MemoryInfoDeviceType.CPU is not None
        assert ort.MemoryInfoDeviceType.GPU is not None
        assert ort.MemoryInfoDeviceType.FPGA is not None
        assert ort.MemoryInfoDeviceType.NPU is not None
