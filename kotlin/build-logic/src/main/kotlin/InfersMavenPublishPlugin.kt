import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.credentials.PasswordCredentials
import org.gradle.api.publish.PublishingExtension
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.credentials

/**
 * Applies vanniktech maven-publish and registers the GitHub Packages repository.
 *
 * Version is `VERSION_NAME` (see root `build.gradle.kts` / `gradle.properties`).
 *
 * Auth: `GITHUB_ACTOR` + `GITHUB_TOKEN` (Actions provides both; locally use a PAT
 * with `write:packages`). Repository path defaults to `stashymane/infe.rs`, or
 * `GITHUB_REPOSITORY` when set.
 */
class InfersMavenPublishPlugin : Plugin<Project> {
    override fun apply(target: Project) {
        target.pluginManager.apply("com.vanniktech.maven.publish")

        val repo: String = System.getenv("GITHUB_REPOSITORY") ?: return

        target.extensions.configure<PublishingExtension> {
            repositories.maven {
                name = "GitHubPackages"
                url = target.uri("https://maven.pkg.github.com/$repo")

                credentials(PasswordCredentials::class) {
                    username = target.providers.environmentVariable("GITHUB_ACTOR").getOrElse("")
                    password = target.providers.environmentVariable("GITHUB_TOKEN").getOrElse("")
                }
            }
        }
    }
}
