import com.android.build.api.dsl.LibraryExtension
import org.gradle.api.JavaVersion
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.artifacts.VersionCatalogsExtension
import org.gradle.api.tasks.testing.Test
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.getByType
import org.gradle.kotlin.dsl.withType
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension

/**
 * Shared KMP library defaults.
 *
 * Android modules must set `android.namespace` themselves.
 */
class InfersKmpLibraryPlugin : Plugin<Project> {
    override fun apply(target: Project) {
        target.pluginManager.apply("org.jetbrains.kotlin.multiplatform")
        target.pluginManager.apply("com.android.library")

        val catalog = target.extensions.getByType<VersionCatalogsExtension>().named("libs")
        val compileSdkVersion = catalog.findVersion("android-compile-sdk").get().requiredVersion.toInt()
        val minSdkVersion = catalog.findVersion("android-min-sdk").get().requiredVersion.toInt()
        val jvmTargetVersion = catalog.findVersion("jvm-target").get().requiredVersion

        target.extensions.configure<KotlinMultiplatformExtension> {
            androidTarget {
                compilerOptions {
                    jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion))
                }
            }
            jvm {
                compilerOptions {
                    jvmTarget.set(JvmTarget.fromTarget(jvmTargetVersion))
                }
            }

            compilerOptions {
                optIn.add("kotlinx.coroutines.ExperimentalCoroutinesApi")
            }

            sourceSets.getByName("commonTest").dependencies {
                implementation(catalog.findLibrary("kotlin-test").get())
                implementation(catalog.findLibrary("kotlinx-coroutines-test").get())
            }

            sourceSets.matching { it.name == "androidInstrumentedTest" }.configureEach {
                dependencies {
                    implementation(catalog.findLibrary("androidx-test-ext-junit").get())
                    implementation(catalog.findLibrary("androidx-test-runner").get())
                    implementation(catalog.findLibrary("kotlin-test").get())
                    implementation(catalog.findLibrary("kotlinx-coroutines-test").get())
                }
            }
        }

        target.extensions.configure<LibraryExtension> {
            compileSdk = compileSdkVersion
            defaultConfig {
                minSdk = minSdkVersion
                testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
                // ExecuTorch Android static libs are built for arm64-v8a only
                // (see scripts/build_executorch.sh android-arm64). Gobley keys
                // cargo Android ABIs off ndk.abiFilters.
                ndk.abiFilters += "arm64-v8a"
            }
            compileOptions {
                sourceCompatibility = JavaVersion.toVersion(jvmTargetVersion)
                targetCompatibility = JavaVersion.toVersion(jvmTargetVersion)
            }
        }

        target.tasks.withType<Test>().configureEach {
            jvmArgs("--enable-native-access=ALL-UNNAMED")
        }
    }
}
