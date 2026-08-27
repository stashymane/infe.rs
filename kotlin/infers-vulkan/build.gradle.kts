plugins {
    id("infers.kmp-feature")
    id("infers.maven-publish")
}

android {
    namespace = "dev.stashy.infers.vulkan"
}

infersCargoFeature {
    name = "vulkan"
}
