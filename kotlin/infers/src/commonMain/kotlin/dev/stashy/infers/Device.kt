package dev.stashy.infers

public enum class DeviceKind {
    Cpu,
    Gpu,
    Npu,
}

/** Device identity metadata (kind, index, name). */
public data class DeviceInfo(
    public val kind: DeviceKind,
    public val id: ULong,
    public val name: String,
)

/**
 * Live execution device handle. [CpuDevice] and [dev.stashy.infers.vulkan.GpuDevice] are the
 * concrete types. Out-of-tree implementors require [InfersInternalApi] opt-in.
 */
@SubclassOptInRequired(InfersInternalApi::class)
public interface Device {
    public val info: DeviceInfo
}

/** Singleton CPU execution device. */
public object CpuDevice : Device {
    override val info: DeviceInfo = DeviceInfo(DeviceKind.Cpu, 0u, "CPU")
}
