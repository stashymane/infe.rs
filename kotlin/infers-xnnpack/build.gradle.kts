plugins {
    id("infers.kmp-feature")
    alias(libs.plugins.maven.publish)
}

android {
    namespace = "dev.stashy.infers.xnnpack"
}

infersCargoFeature {
    name = "xnnpack"
}
