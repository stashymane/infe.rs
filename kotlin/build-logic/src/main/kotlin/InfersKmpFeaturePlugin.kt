import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.withType
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.tasks.KotlinCompilationTask

/**
 * Convention for Cargo-feature KMP modules (`:infers-xnnpack`, `:infers-vulkan`, …).
 *
 * Applies library + ktlint + explicit API + cargo-feature, wires
 * `api(project(":infers"))` + `implementation(project(":infers-ffi"))`,
 * opts in to `InfersInternalApi`, and makes Kotlin compile tasks depend on
 * `checkInfersCargoFeature`.
 */
class InfersKmpFeaturePlugin : Plugin<Project> {
    override fun apply(target: Project) {
        target.pluginManager.apply("infers.kmp-library")
        target.pluginManager.apply("infers.ktlint")
        target.pluginManager.apply("infers.explicit-api")
        target.pluginManager.apply("infers.cargo-feature")

        target.extensions.configure<KotlinMultiplatformExtension> {
            compilerOptions {
                optIn.add("dev.stashy.infers.InfersInternalApi")
            }
            sourceSets.getByName("commonMain").dependencies {
                api(project(":infers"))
                implementation(project(":infers-ffi"))
            }
        }

        val checkFeature = target.tasks.named("checkInfersCargoFeature")
        target.tasks.withType<KotlinCompilationTask<*>>().configureEach {
            dependsOn(checkFeature)
        }

    }
}
