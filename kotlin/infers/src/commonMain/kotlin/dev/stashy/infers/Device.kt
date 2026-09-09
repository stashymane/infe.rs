package dev.stashy.infers

/**
 * Live execution device handle.
 */
@SubclassOptInRequired(InfersInternalApi::class)
public interface Device<Self : Device<Self>> {
    public val info: DeviceInfo

    /** Place [hardware] on this device without committing transfer/compute. */
    @InfersInternalApi
    public fun deferHardwareInternal(hardware: HardwareImage): Deferred<Self>

    /**
     * Place a pooled [frame] on this device without committing transfer/compute.
     *
     * Default: copy any imported host-visible buffer into [FrameBuffer.host], then
     * [deferHardwareInternal]. GPU backends may override for zero-copy imports.
     */
    @InfersInternalApi
    public fun deferFrameInternal(frame: FrameBuffer<Self>): Deferred<Self> {
        frame.prepareHostForDefer()
        return deferHardwareInternal(frame.host)
    }
}

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

/** Singleton CPU execution device. */
public object CpuDevice : Device<CpuDevice> {
    override val info: DeviceInfo = DeviceInfo(DeviceKind.Cpu, 0u, "CPU")

    @OptIn(InfersInternalApi::class)
    override fun deferHardwareInternal(hardware: HardwareImage): Deferred<CpuDevice> {
        hardware.ensureOpen()
        return CpuDeferred(hardware.handle.onCpu())
    }
}
