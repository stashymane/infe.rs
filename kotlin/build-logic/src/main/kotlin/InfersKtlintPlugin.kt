import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.artifacts.VersionCatalogsExtension
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.getByType
import org.jlleitschuh.gradle.ktlint.KtlintExtension

/**
 * Applies ktlint with the CLI version from the version catalog.
 *
 * ktlint-gradle's default CLI is still 1.5.x; context parameters need >= 1.7
 * (`ktlint-cli` in `libs.versions.toml`).
 */
class InfersKtlintPlugin : Plugin<Project> {
    override fun apply(target: Project) {
        target.pluginManager.apply("org.jlleitschuh.gradle.ktlint")

        val catalog = target.extensions.getByType<VersionCatalogsExtension>().named("libs")
        val ktlintCli = catalog.findVersion("ktlint-cli").get().requiredVersion
        target.extensions.configure<KtlintExtension> {
            version.set(ktlintCli)
        }
    }
}
