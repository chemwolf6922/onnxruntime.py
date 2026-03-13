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

    def test_default_device_mem_type(self):
        mem = ort.MemoryInfo()
        assert mem.device_mem_type == ort.DeviceMemoryType.DEFAULT

    def test_default_vendor_id(self):
        mem = ort.MemoryInfo()
        assert isinstance(mem.vendor_id, int)


class TestMemoryInfoCreate:
    def test_create(self):
        mem = ort.MemoryInfo.create("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        assert mem.name == "Cpu"
        assert mem.allocator_type == ort.AllocatorType.ARENA
        assert mem.device_id == 0
        assert mem.mem_type == ort.MemType.DEFAULT

    def test_create_cpu_input_mem_type(self):
        mem = ort.MemoryInfo.create("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.CPU_INPUT)
        assert mem.mem_type == ort.MemType.CPU_INPUT

    def test_create_cpu_output_mem_type(self):
        mem = ort.MemoryInfo.create("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.CPU_OUTPUT)
        assert mem.mem_type == ort.MemType.CPU_OUTPUT

    def test_create_device_mem_type(self):
        mem = ort.MemoryInfo.create("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        assert mem.device_mem_type == ort.DeviceMemoryType.DEFAULT

    def test_create_vendor_id(self):
        mem = ort.MemoryInfo.create("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        assert isinstance(mem.vendor_id, int)


class TestMemoryInfoCreateV2:
    def test_create_v2(self):
        mem = ort.MemoryInfo.create_v2(
            "Cpu",
            ort.MemoryInfoDeviceType.CPU,
            0,  # vendor_id
            0,  # device_id
            ort.DeviceMemoryType.DEFAULT,
            0,  # alignment
            ort.AllocatorType.DEVICE,
        )
        assert mem.name == "Cpu"
        assert mem.device_type == ort.MemoryInfoDeviceType.CPU
        assert mem.vendor_id == 0
        assert mem.device_id == 0
        assert mem.device_mem_type == ort.DeviceMemoryType.DEFAULT
        assert mem.allocator_type == ort.AllocatorType.DEVICE

    def test_create_v2_host_accessible(self):
        mem = ort.MemoryInfo.create_v2(
            "Cpu",
            ort.MemoryInfoDeviceType.CPU,
            0,
            0,
            ort.DeviceMemoryType.HOST_ACCESSIBLE,
            0,
            ort.AllocatorType.DEVICE,
        )
        assert mem.device_mem_type == ort.DeviceMemoryType.HOST_ACCESSIBLE


class TestMemoryInfoComparison:
    def test_equal_default(self):
        a = ort.MemoryInfo()
        b = ort.MemoryInfo()
        assert a == b

    def test_equal_create_same(self):
        a = ort.MemoryInfo.create("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        b = ort.MemoryInfo.create("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        assert a == b

    def test_not_equal_different_allocator(self):
        a = ort.MemoryInfo.create("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.DEFAULT)
        b = ort.MemoryInfo.create("Cpu", ort.AllocatorType.ARENA, 0, ort.MemType.DEFAULT)
        assert not (a == b)

    def test_not_equal_different_mem_type(self):
        a = ort.MemoryInfo.create("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.DEFAULT)
        b = ort.MemoryInfo.create("Cpu", ort.AllocatorType.DEVICE, 0, ort.MemType.CPU_INPUT)
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

    def test_device_memory_type_values(self):
        assert ort.DeviceMemoryType.DEFAULT is not None
        assert ort.DeviceMemoryType.HOST_ACCESSIBLE is not None
