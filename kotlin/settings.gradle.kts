pluginManagement {
    includeBuild("build-logic")
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

@Suppress("UnstableApiUsage")
dependencyResolutionManagement {
    repositoriesMode = RepositoriesMode.FAIL_ON_PROJECT_REPOS
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "infers-kotlin"

include(
    ":infers",
    ":infers-ffi",
    ":infers-portable",
    ":infers-xnnpack",
    ":infers-vulkan",
    ":infers-consumer-test",
)
