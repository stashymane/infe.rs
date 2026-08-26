import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.kotlin.dsl.configure
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation

/**
 * Enables Kotlin `explicitApi()` and KGP ABI validation for publishable modules.
 */
class InfersExplicitApiPlugin : Plugin<Project> {
    override fun apply(target: Project) {
        target.pluginManager.withPlugin("org.jetbrains.kotlin.multiplatform") {
            target.extensions.configure<KotlinMultiplatformExtension> {
                explicitApi()
                @OptIn(ExperimentalAbiValidation::class)
                abiValidation()
            }
        }
    }
}
