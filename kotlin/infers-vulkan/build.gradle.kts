plugins {
    id("infers.kmp-library")
    id("infers.ktlint")
    id("infers.explicit-api")
    id("infers.maven-publish")
}

android {
    namespace = "dev.stashy.infers.vulkan"
}

kotlin {
    compilerOptions {
        optIn.add("dev.stashy.infers.InfersInternalApi")
    }
    sourceSets {
        commonMain.dependencies {
            api(projects.infers)
            implementation(projects.infersFfi)
        }
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
