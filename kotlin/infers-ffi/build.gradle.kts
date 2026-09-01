import gobley.gradle.GobleyHost
import gobley.gradle.cargo.dsl.android
import gobley.gradle.cargo.dsl.jvm
import gobley.gradle.cargo.tasks.CargoBuildTask
import gobley.gradle.rust.targets.RustAndroidTarget
import gobley.gradle.rust.targets.RustPosixTarget
import gobley.gradle.rust.targets.RustTarget

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

val androidApi: Int = libs.versions.android.min.sdk
    .get()
    .toInt()

val repoRoot = rootProject.layout.projectDirectory.dir("..")
val executorchOutputDir = repoRoot.dir("target/executorch")

// Prebuilt ExecuTorch trees from scripts/build_executorch.sh / CI artifacts.
val executorchLibDirs: Map<RustTarget, String> = mapOf(
    RustAndroidTarget.Arm64 to executorchOutputDir.dir("android-arm64").asFile.absolutePath,
    RustPosixTarget.LinuxX64 to executorchOutputDir.dir("x86_64-unknown-linux-gnu").asFile.absolutePath,
)

cargo {
    packageDirectory = rootProject.layout.projectDirectory.dir("../crates/bindings")
    features = setOf(
        "portable",
        "xnnpack",
        "vulkan",
    )
    builds.jvm {
        embedRustLibrary = rustTarget == GobleyHost.current.rustTarget
    }
    builds.android {
        // NDK C++ runtime required by libinfers_bindings.so (and ET/Vulkan deps).
        dynamicLibraries.add("c++_shared")
    }
    builds.configureEach {
        val libDir = executorchLibDirs[rustTarget] ?: return@configureEach
        variants {
            buildTaskProvider.configure {
                // Gobley cargo tasks use a curated env; an empty EXECUTORCH_RS_EXECUTORCH_LIB_DIR
                // blocks the .cargo/config.toml fallback.
                additionalEnvironment.put("EXECUTORCH_RS_EXECUTORCH_LIB_DIR", libDir)
            }
        }
    }
}

uniffi {
    generateFromLibrary {
        namespace = "infers_bindings"
        packageName = "dev.stashy.infers.ffi"
        // Generate from the host library so JVM compiles don't require a successful Android build first.
        build = GobleyHost.current.rustTarget
        variant = gobley.gradle.Variant.Debug
    }
}

tasks.withType<CargoBuildTask>().configureEach {
    // Gradle owns the feature set; do not also pull Cargo.toml defaults.
    extraArguments.add("--no-default-features")
}

// Gobley sets BINDGEN_EXTRA_CLANG_ARGS_<triple> (hyphens) to --sysroot only.
// bindgen prefers that hyphen form over the underscore variant, and NDK r26+
// requires an explicit --target=<triple><api>.
// Merge after Gobley has configured NDK env — additionalEnvironment is finalized
// before task actions, so doFirst mutation fails on Gradle 9+.
afterEvaluate {
    cargo.builds.android.configureEach {
        val triple = rustTarget.rustTriple
        variants {
            buildTaskProvider.configure {
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
            }
        }
    }
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(libs.kotlinx.coroutines.core)
            implementation(libs.atomicfu)
        }
        jvmMain.dependencies {
            implementation(libs.jna)
        }
    }
}
