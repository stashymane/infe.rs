#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-/workspace}"
KOTLIN_DIR="${ROOT}/kotlin"

if [[ ! -d "${KOTLIN_DIR}" ]]; then
    echo "Kotlin project not found at ${KOTLIN_DIR}. Mount the repo at /workspace." >&2
    exit 1
fi

if [[ -z "${ANDROID_HOME:-}" || ! -d "${ANDROID_HOME}" ]]; then
    echo "ANDROID_HOME is not set or missing." >&2
    exit 1
fi

# Prefer an injected local.properties (scripts/android_gradle.sh bind-mounts one).
# Fall back to writing sdk.dir only when the file is absent or writable.
if [[ ! -f "${KOTLIN_DIR}/local.properties" ]]; then
    printf 'sdk.dir=%s\n' "${ANDROID_HOME}" > "${KOTLIN_DIR}/local.properties"
elif [[ -w "${KOTLIN_DIR}/local.properties" ]]; then
    printf 'sdk.dir=%s\n' "${ANDROID_HOME}" > "${KOTLIN_DIR}/local.properties"
fi

export ANDROID_SDK_ROOT="${ANDROID_HOME}"
export GRADLE_USER_HOME="${GRADLE_USER_HOME:-/root/.gradle}"
export CARGO_HOME="${CARGO_HOME:-/opt/cargo}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${ROOT}/target}"
export PATH="${ANDROID_HOME}/platform-tools:${PATH}"

cd "${KOTLIN_DIR}"

if [[ "$#" -eq 0 ]]; then
    set -- :infers:connectedDebugAndroidTest :infers-vulkan:connectedDebugAndroidTest --console=plain
fi

echo "adb devices:"
adb devices -l || true
echo "Running: ./gradlew $*"

status=0
./gradlew "$@" || status=$?

# Rootful Docker writes as root into the bind-mounted workspace; restore ownership
# so host IDE Gradle sync can rewrite kotlin/**/build. Skip under user namespaces
# where container uid 0 already maps to the host user.
if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
    host_uid_for_0="$(awk '$1 == 0 { print $2; exit }' /proc/self/uid_map 2>/dev/null || true)"
    if [[ "${host_uid_for_0}" == "0" ]]; then
        chown -R "${HOST_UID}:${HOST_GID}" \
            "${CARGO_TARGET_DIR}" \
            "${KOTLIN_DIR}/.kotlin" \
            "${KOTLIN_DIR}/build" \
            "${KOTLIN_DIR}"/*/build \
            2>/dev/null || true
    fi
fi

exit "${status}"
