plugins {
    id("infers.kmp-library")
    id("infers.ktlint")
}

android {
    namespace = "dev.stashy.infers.consumer.test"
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(project(":infers"))
        }
    }
}
