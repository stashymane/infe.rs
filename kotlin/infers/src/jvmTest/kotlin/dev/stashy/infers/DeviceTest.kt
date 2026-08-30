package dev.stashy.infers

import kotlin.test.Test
import kotlin.test.assertEquals

class DeviceTest {
    @Test
    fun cpuDeviceFactory() {
        val device = Device.cpu()
        assertEquals(DeviceKind.Cpu, device.kind)
        assertEquals(0u, device.id)
        assertEquals("CPU", device.name)
    }

    @Test
    fun gpuDeviceFactory() {
        val device = Device.gpu(0u)
        assertEquals(DeviceKind.Gpu, device.kind)
        assertEquals(0u, device.id)
        assertEquals("GPU:0", device.name)
    }

    @Test
    fun npuDeviceFactory() {
        val device = Device.npu(0u)
        assertEquals(DeviceKind.Npu, device.kind)
        assertEquals(0u, device.id)
        assertEquals("NPU:0", device.name)
    }
}
