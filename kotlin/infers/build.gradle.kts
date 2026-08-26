plugins {
    id("infers.kmp-library")
    id("infers.ktlint")
    id("infers.explicit-api")
    alias(libs.plugins.maven.publish)
}

android {
    namespace = "dev.stashy.infers.core"
}

kotlin {
    compilerOptions {
        optIn.add("dev.stashy.infers.InfersInternalApi")
    }
    sourceSets {
        commonMain.dependencies {
            implementation(project(":infers-ffi"))
            api(libs.kotlinx.coroutines.core)
            api(libs.kotlinx.io.core)
        }
        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}
