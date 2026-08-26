package dev.stashy.infers.portable

/**
 * Marker for the ExecuTorch `portable` kernels Cargo feature.
 *
 * Depending on `:infers-portable` selects the feature at compile time; there is
 * no additional UniFFI surface beyond what `:infers` already exposes.
 */
public object InfersPortable
