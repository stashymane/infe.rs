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
        register("infersCargoFeature") {
            id = "infers.cargo-feature"
            implementationClass = "InfersCargoFeaturePlugin"
        }
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
        register("infersKmpFeature") {
            id = "infers.kmp-feature"
            implementationClass = "InfersKmpFeaturePlugin"
        }
        register("infersMavenPublish") {
            id = "infers.maven-publish"
            implementationClass = "InfersMavenPublishPlugin"
        }
    }
}
