plugins {
    id("infers.kmp-feature")
    alias(libs.plugins.maven.publish)
}

android {
    namespace = "dev.stashy.infers.vulkan"
}

infersCargoFeature {
    name = "vulkan"
}
