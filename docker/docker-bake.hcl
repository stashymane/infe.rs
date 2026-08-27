# Bake file for infers Docker images.
# Usage (from this directory):
#   docker buildx bake
#   docker buildx bake builder
#   docker buildx bake android
#
# Local context paths are relative to the process CWD. Prefer running bake from
# docker/, as the project scripts do.

group "default" {
  targets = ["base", "builder", "android"]
}

target "_common" {
  dockerfile = "Dockerfile"
  # Load into the local engine so scripts can `docker run` / `docker image inspect`.
  output = ["type=docker"]
}

target "base" {
  inherits = ["_common"]
  context = "./base"
  tags = ["infers-base:latest"]
}

target "builder" {
  inherits = ["_common"]
  context = "./builder"
  contexts = {
    infers-base = "target:base"
  }
  tags = ["infers-builder:latest"]
}

target "android" {
  inherits = ["_common"]
  context = "./android"
  contexts = {
    infers-base = "target:base"
  }
  tags = ["infers-android-builder:latest"]
}
