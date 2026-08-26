package dev.stashy.infers

/**
 * Marks APIs that exist only so feature modules can talk to the generated FFI
 * layer. App code must not use these; they require an explicit opt-in and the
 * `:infers-ffi` dependency, which apps never declare.
 */
@RequiresOptIn(
    level = RequiresOptIn.Level.ERROR,
    message = "Infers internal bridge API — for Infers feature modules only",
)
@Retention(AnnotationRetention.BINARY)
@Target(
    AnnotationTarget.CLASS,
    AnnotationTarget.FUNCTION,
    AnnotationTarget.PROPERTY,
    AnnotationTarget.CONSTRUCTOR,
    AnnotationTarget.TYPEALIAS,
)
public annotation class InfersInternalApi
