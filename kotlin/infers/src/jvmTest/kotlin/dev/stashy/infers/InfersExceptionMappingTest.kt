package dev.stashy.infers

import kotlin.test.Test
import kotlin.test.assertEquals

class InfersExceptionMappingTest {
    @Test
    fun mapsAllFfiVariants() {
        val cases =
            listOf(
                dev.stashy.infers.ffi.InfersException
                    .InvalidShape("boom") to InfersException.InvalidShape::class,
                dev.stashy.infers.ffi.InfersException
                    .UnsupportedType("boom") to InfersException.UnsupportedType::class,
                dev.stashy.infers.ffi.InfersException
                    .BufferAllocationFailed("boom") to
                    InfersException.BufferAllocationFailed::class,
                dev.stashy.infers.ffi.InfersException
                    .ModelLoadFailed("boom") to InfersException.ModelLoadFailed::class,
                dev.stashy.infers.ffi.InfersException
                    .InferenceFailed("boom") to InfersException.InferenceFailed::class,
                dev.stashy.infers.ffi.InfersException
                    .ProcessingFailed("boom") to InfersException.ProcessingFailed::class,
                dev.stashy.infers.ffi.InfersException
                    .PlatformException("boom") to InfersException.PlatformError::class,
                dev.stashy.infers.ffi.InfersException
                    .InternalException("boom") to InfersException.InternalError::class,
            )
        for ((ffi, expected) in cases) {
            val mapped =
                dev.stashy.infers.internal
                    .mapFfiException(ffi)
            assertEquals(expected, mapped::class)
            assertEquals("boom", mapped.reason)
        }

        val mismatch =
            dev.stashy.infers.internal.mapFfiException(
                dev.stashy.infers.ffi.InfersException
                    .DeviceMismatch("cpu", "gpu"),
            ) as InfersException.DeviceMismatch
        assertEquals("cpu", mismatch.expected)
        assertEquals("gpu", mismatch.actual)
        assertEquals("expected cpu, actual gpu", mismatch.reason)
    }
}
