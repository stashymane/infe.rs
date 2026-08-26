import org.gradle.api.Project
import org.gradle.api.provider.Provider

/** Resolves the comma-separated `infers.cargo.features` Gradle property. */
object InfersCargoFeatures {
    private const val PROPERTY = "infers.cargo.features"

    fun provider(project: Project): Provider<Set<String>> =
        project.providers.gradleProperty(PROPERTY).map(::parse).orElse(emptySet())

    fun enabled(project: Project): Set<String> = provider(project).get()

    fun parse(raw: String): Set<String> =
        raw.split(',')
            .map(String::trim)
            .filter(String::isNotEmpty)
            .toSet()
}
