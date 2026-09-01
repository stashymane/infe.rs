package dev.stashy.infers

import kotlin.test.Test
import kotlin.test.assertEquals

class DeviceTest {
    @Test
    fun cpuDeviceSingleton() {
        assertEquals(DeviceKind.Cpu, CpuDevice.info.kind)
        assertEquals(0u, CpuDevice.info.id)
        assertEquals("CPU", CpuDevice.info.name)
    }

    @Test
    fun deviceInfoRecord() {
        val info = DeviceInfo(DeviceKind.Gpu, 0u, "GPU:0")
        assertEquals(DeviceKind.Gpu, info.kind)
        assertEquals(0u, info.id)
        assertEquals("GPU:0", info.name)
    }
}
