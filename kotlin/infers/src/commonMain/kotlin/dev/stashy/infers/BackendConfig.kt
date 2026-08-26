package dev.stashy.infers

/**
 * Opaque backend configuration. Concrete variants live in feature modules
 * (`XnnpackConfig`, `VulkanConfig`, …). Core declares no variants of its own.
 *
 * Feature modules implement [dev.stashy.infers.internal.BackendConfigFfiConvertible]
 * so [dev.stashy.infers.internal.toFfi] can convert without a global registry.
 */
public interface BackendConfig
