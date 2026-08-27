# Bake file for infers Docker images.
# Usage (from this directory):
#   docker buildx bake
#   docker buildx bake builder
#   docker buildx bake android
#
# Local context paths are relative to the process CWD. Prefer running bake from
# docker/, as the project scripts do.
#
# CI merges docker/metadata-action bake files that populate the empty
# docker-metadata-action-* targets (tags/labels). See publish-kotlin.yml.

group "default" {
  targets = ["base", "builder", "android"]
}

target "_common" {
  dockerfile = "Dockerfile"
  # Load into the local engine so scripts can `docker run` / `docker image inspect`.
  output = ["type=docker"]
}

# Placeholders overridden by docker/metadata-action bake files in GHA.
target "docker-metadata-action-builder" {}
target "docker-metadata-action-android" {}

target "base" {
  inherits = ["_common"]
  context = "./base"
  tags = ["infers-base:latest"]
}

target "builder" {
  inherits = ["_common", "docker-metadata-action-builder"]
  context = "./builder"
  contexts = {
    infers-base = "target:base"
  }
  tags = ["infers-builder:latest"]
}

target "android" {
  inherits = ["_common", "docker-metadata-action-android"]
  context = "./android"
  contexts = {
    infers-base = "target:base"
  }
  tags = ["infers-android-builder:latest"]
}
