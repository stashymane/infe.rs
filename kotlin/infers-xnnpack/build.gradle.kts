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

kotlin {
    sourceSets {
        jvmTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
        androidInstrumentedTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
            implementation(libs.androidx.test.runner)
        }
    }
}
