import org.gradle.api.DefaultTask
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.provider.Property
import org.gradle.api.provider.SetProperty
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.TaskAction
import org.gradle.kotlin.dsl.create
import org.gradle.kotlin.dsl.register

abstract class InfersCargoFeatureExtension {
    abstract val name: Property<String>
}

abstract class CheckInfersCargoFeature : DefaultTask() {
    @get:Input
    abstract val modulePath: Property<String>

    @get:[Input Optional]
    abstract val feature: Property<String>

    @get:Input
    abstract val enabledFeatures: SetProperty<String>

    @TaskAction
    fun verify() {
        val module = modulePath.get()
        val required = feature.orNull
            ?: error("$module: set infersCargoFeature { name = \"...\" }")
        val enabled = enabledFeatures.get()
        check(required in enabled) {
            "$module requires Cargo feature '$required', but " +
                "infers.cargo.features=${enabled.joinToString(",")}. " +
                "Add '$required' to kotlin/gradle.properties (or -Pinfers.cargo.features=...)."
        }
    }
}

/**
 * Marks a Gradle module as a Cargo feature selector for the single `infers_bindings` library.
 *
 * Registers `checkInfersCargoFeature` to verify the feature is listed in `infers.cargo.features`.
 */
class InfersCargoFeaturePlugin : Plugin<Project> {
    override fun apply(target: Project) {
        val extension = target.extensions.create<InfersCargoFeatureExtension>("infersCargoFeature")

        val checkFeature = target.tasks.register<CheckInfersCargoFeature>("checkInfersCargoFeature") {
            group = "verification"
            description = "Verifies this module's Cargo feature is listed in infers.cargo.features"
            modulePath.set(target.path)
            feature.set(extension.name)
            enabledFeatures.set(InfersCargoFeatures.provider(target))
        }

        target.tasks.named("check") {
            dependsOn(checkFeature)
        }
    }
}
