# Consumer R8/ProGuard rules for Android-specific JNI in the core library.

# JNI: Java_dev_stashy_infers_HardwareBufferBridge_nativePointer
-keep class dev.stashy.infers.HardwareBufferBridge {
    native <methods>;
}
