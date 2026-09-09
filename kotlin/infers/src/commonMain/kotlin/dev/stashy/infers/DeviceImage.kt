package dev.stashy.infers

/** Device-resident image input for preprocessing on [D]. */
public interface DeviceImage<D : Device<D>> : AutoCloseable {
    public val width: UInt
    public val height: UInt
    public val format: ImageFormat
}
