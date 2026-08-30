plugins {
    id("infers.kmp-library")
    id("infers.ktlint")
    id("infers.explicit-api")
    id("infers.maven-publish")
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
            implementation(projects.infersFfi)
            api(libs.kotlinx.coroutines.core)
            api(libs.kotlinx.io.core)
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
