package dev.stashy.infers

import kotlin.test.Test
import kotlin.test.assertEquals

class ProcessingOptionsBuilderTest {
    @Test
    fun buildsFullFrameFromSourceAndDest() {
        val options = ProcessingOptions {
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
        val options = ProcessingOptions {
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
