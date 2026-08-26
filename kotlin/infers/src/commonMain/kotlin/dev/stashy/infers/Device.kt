package dev.stashy.infers

public enum class DeviceKind {
    Cpu,
    Gpu,
    Npu,
}

public data class Device(
    public val kind: DeviceKind,
    public val id: ULong,
    public val name: String,
) {
    public companion object {
        public fun cpu(): Device = Device(DeviceKind.Cpu, 0u, "CPU")

        public fun gpu(id: ULong = 0u): Device = Device(DeviceKind.Gpu, id, "GPU:$id")

        public fun npu(id: ULong = 0u): Device = Device(DeviceKind.Npu, id, "NPU:$id")
    }
}
