plugins {
    `kotlin-dsl`
}

repositories {
    google()
    mavenCentral()
    gradlePluginPortal()
}

dependencies {
    implementation(libs.kotlin.gradle.plugin)
    implementation(libs.android.gradle.plugin)
    implementation(libs.ktlint.gradle)
    implementation(libs.maven.publish.gradle.plugin)
}

gradlePlugin {
    plugins {
        register("infersKmpLibrary") {
            id = "infers.kmp-library"
            implementationClass = "InfersKmpLibraryPlugin"
        }
        register("infersKtlint") {
            id = "infers.ktlint"
            implementationClass = "InfersKtlintPlugin"
        }
        register("infersExplicitApi") {
            id = "infers.explicit-api"
            implementationClass = "InfersExplicitApiPlugin"
        }
        register("infersMavenPublish") {
            id = "infers.maven-publish"
            implementationClass = "InfersMavenPublishPlugin"
        }
    }
}
