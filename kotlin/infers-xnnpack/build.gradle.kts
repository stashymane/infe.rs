plugins {
    id("infers.kmp-feature")
    id("infers.maven-publish")
}

android {
    namespace = "dev.stashy.infers.xnnpack"
}

infersCargoFeature {
    name = "xnnpack"
}
