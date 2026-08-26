package dev.stashy.infers

import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

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

class TensorTest {
    @Test
    fun floatTensorRoundTrip() =
        runTest {
            inferenceScope {
                val shape = TensorShape.of(1, 3, 2, 2)
                val data = FloatArray(12) { it.toFloat() }
                val tensor = tensorOf(shape, data)
                assertEquals(shape, tensor.shape)
                assertEquals(DataType.F32, tensor.dtype)
                assertEquals(48u, tensor.byteSize)
                assertTrue(tensor.readFloats().contentEquals(data))

                val copied = tensor.copyTo(Device.cpu())
                assertTrue(copied.readFloats().contentEquals(data))
            }
        }

    @Test
    fun closedTensorThrows() =
        runTest {
            val tensor =
                inferenceScope {
                    tensorOf(TensorShape.of(1), floatArrayOf(1f)).also {
                        // Escape intentionally — scope will close it.
                    }
                }
            assertFailsWith<IllegalStateException> {
                tensor.readFloats()
            }
        }
}

class CpuImageProcessorTest {
    @Test
    fun resizeRgb888ToRgbf32() =
        runTest {
            val rawBytes = ByteArray(48) { (it * 5).toByte() }
            CpuImageProcessor().use { processor ->
                assertEquals(DeviceKind.Cpu, processor.device.kind)
                inferenceScope {
                    val out = processor.process(rawBytes, TestFixtures.rgb4x4To2x2())
                    assertEquals(TensorShape.of(1, 2, 2, 3), out.shape)
                    assertEquals(DataType.F32, out.dtype)
                    assertEquals(12, out.readFloats().size)
                }
            }
        }
}

class InferenceScopeTest {
    @Test
    fun closesResourcesWhenBlockThrows() =
        runTest {
            assertFailsWith<IllegalStateException> {
                inferenceScope {
                    tensorOf(TensorShape.of(1), floatArrayOf(1f))
                    error("boom")
                }
            }
            val escaped =
                inferenceScope {
                    tensorOf(TensorShape.of(1), floatArrayOf(2f))
                }
            assertFailsWith<IllegalStateException> {
                escaped.readBytes()
            }
        }

    @Test
    fun closesEvenWhenBlockSucceeds() =
        runTest {
            val tensor =
                inferenceScope {
                    tensorOf(TensorShape.of(2), floatArrayOf(1f, 2f))
                }
            assertFailsWith<IllegalStateException> {
                tensor.readFloats()
            }
        }
}

class MapInferenceTest {
    @Test
    fun mapInferenceRunsInsideScope() =
        runTest {
            val results =
                flowOfValues(1, 2, 3)
                    .mapInference { n ->
                        tensorOf(TensorShape.of(1), floatArrayOf(n.toFloat())).readFloats()[0]
                    }.toList()
            assertEquals(listOf(1f, 2f, 3f), results)
        }

    @Test
    fun conflateDropsStaleFramesBeforeMapInference() =
        runTest {
            val collected = mutableListOf<Int>()
            val gate = Channel<Unit>(capacity = Channel.RENDEZVOUS)
            val job =
                launch {
                    flow {
                        emit(1)
                        gate.receive()
                        emit(2)
                        emit(3)
                    }.conflate()
                        .mapInference { it }
                        .collect { value ->
                            collected += value
                            if (value == 1) {
                                gate.send(Unit)
                                delay(1_000)
                            }
                        }
                }
            advanceUntilIdle()
            job.cancel()
            // Collector takes 1, then 2 and 3 arrive while it is delayed; conflate keeps 3.
            assertEquals(listOf(1, 3), collected)
        }

    private fun flowOfValues(vararg values: Int) = flow { values.forEach { emit(it) } }
}

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
            val mapped = dev.stashy.infers.internal.mapFfiException(ffi)
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

internal object TestFixtures {
    fun rgb4x4To2x2(): ProcessingOptions =
        ProcessingOptions {
            source = 4 to 4
            dest = 2 to 2
            srcFormat = ImageFormat.Rgb888
            destFormat = ImageFormat.Rgbf32
        }
}

class ProcessingOptionsBuilderTest {
    @Test
    fun buildsFullFrameFromSourceAndDest() {
        val options =
            ProcessingOptions {
                source = 1920 to 1080
                dest = 300 to 300
                srcFormat = ImageFormat.Rgb888
                destFormat = ImageFormat.Rgbf32
                fitMode = FitMode.Contain
                rotation = Rotation.Rot90
            }
        assertEquals(1920u, options.srcW)
        assertEquals(1080u, options.srcH)
        assertEquals(0u, options.cropX)
        assertEquals(0u, options.cropY)
        assertEquals(1920u, options.cropW)
        assertEquals(1080u, options.cropH)
        assertEquals(300u, options.destW)
        assertEquals(300u, options.destH)
        assertEquals(ImageFormat.Rgb888, options.srcFormat)
        assertEquals(ImageFormat.Rgbf32, options.destFormat)
        assertEquals(FitMode.Contain, options.fitMode)
        assertEquals(Rotation.Rot90, options.rotation)
    }

    @Test
    fun respectsExplicitCrop() {
        val options =
            ProcessingOptions {
                source = 100 to 80
                dest = 50 to 40
                cropOrigin = 10 to 5
                cropSize = 60 to 50
                srcFormat = ImageFormat.Nv12
                destFormat = ImageFormat.Rgb888
            }
        assertEquals(10u, options.cropX)
        assertEquals(5u, options.cropY)
        assertEquals(60u, options.cropW)
        assertEquals(50u, options.cropH)
    }
}
