import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi

plugins {
    id("infers.kmp-library")
    id("infers.ktlint")
}

android {
    namespace = "dev.stashy.infers.sample"
}

kotlin {
    compilerOptions {
        optIn.add("dev.stashy.infers.InfersInternalApi")
    }
    jvm {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        binaries {
            executable {
                mainClass.set("dev.stashy.infers.sample.GpuPipelineSampleKt")
            }
        }
    }
    sourceSets {
        jvmMain.dependencies {
            implementation(projects.infersVulkan)
            implementation(libs.kotlinx.coroutines.core)
        }
    }
}

tasks.named<JavaExec>("runJvm") {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
