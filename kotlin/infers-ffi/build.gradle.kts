import gobley.gradle.GobleyHost
import gobley.gradle.Variant
import gobley.gradle.cargo.dsl.android
import gobley.gradle.cargo.dsl.jvm
import gobley.gradle.cargo.tasks.CargoBuildTask

plugins {
    id("infers.kmp-library")
    id("infers.explicit-api")
    alias(libs.plugins.gobley.cargo)
    alias(libs.plugins.gobley.uniffi)
    alias(libs.plugins.kotlin.atomicfu)
    id("infers.maven-publish")
}

android {
    namespace = "dev.stashy.infers.ffi"
}

val cargoFeatures: Set<String> = InfersCargoFeatures.enabled(project)
val androidApi: Int =
    libs.versions.android.min.sdk
        .get()
        .toInt()

cargo {
    packageDirectory = rootProject.layout.projectDirectory.dir("../crates/bindings")
    features = cargoFeatures
    builds.jvm {
        embedRustLibrary = rustTarget == GobleyHost.current.rustTarget
    }
    builds.android {
        // NDK C++ runtime required by libinfers_bindings.so (and ET/Vulkan deps).
        dynamicLibraries.add("c++_shared")
    }
}

uniffi {
    generateFromLibrary {
        namespace = "infers_bindings"
        packageName = "dev.stashy.infers.ffi"
        // Generate from the host library so JVM compiles don't require a successful Android build first.
        build = GobleyHost.current.rustTarget
        variant = Variant.Debug
    }
}

tasks.withType<CargoBuildTask>().configureEach {
    // Gradle owns the feature set; do not also pull Cargo.toml defaults.
    extraArguments.add("--no-default-features")
}

val androidExecuTorchLibs =
    rootProject.layout.projectDirectory
        .dir("../target/executorch/android-arm64")
        .asFile
        .absolutePath

// Gobley sets BINDGEN_EXTRA_CLANG_ARGS_<triple> (hyphens) to --sysroot only.
// bindgen prefers that hyphen form over the underscore variant, and NDK r26+
// requires an explicit --target=<triple><api>.
// Merge after Gobley has configured NDK env — additionalEnvironment is finalized
// before task actions, so doFirst mutation fails on Gradle 9+.
afterEvaluate {
    tasks.withType<CargoBuildTask>().configureEach {
        val triple = target.get().rustTriple
        if (!triple.contains("android")) return@configureEach
        val key = "BINDGEN_EXTRA_CLANG_ARGS_$triple"
        val previous = additionalEnvironment.get()[key]?.toString().orEmpty()
        val targetFlag = "--target=$triple$androidApi"
        val merged =
            when {
                previous.contains("--target=") -> previous
                previous.isBlank() -> targetFlag
                else -> "$targetFlag $previous"
            }
        additionalEnvironment.put(key, merged)
        // Only arm64-v8a has Docker-built ExecuTorch libs; other Android ABIs
        // are filtered out via ndk.abiFilters.
        if (!triple.startsWith("aarch64-")) return@configureEach
        additionalEnvironment.put("EXECUTORCH_RS_EXECUTORCH_LIB_DIR", androidExecuTorchLibs)
    }
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(libs.kotlinx.coroutines.core)
            implementation(libs.atomicfu)
        }
        androidMain.dependencies {
            // Android must use the AAR; the default JAR collides with it on instrumented tests.
            implementation("net.java.dev.jna:jna:${libs.versions.jna.get()}@aar")
        }
        jvmMain.dependencies {
            implementation(libs.jna)
        }
    }
}

tasks.register("printInfersCargoFeatures") {
    group = "help"
    description = "Prints the Cargo features selected for libinfers_bindings"
    val features = cargoFeatures
    doLast {
        println("infers.cargo.features=${features.joinToString(",")}")
    }
}
