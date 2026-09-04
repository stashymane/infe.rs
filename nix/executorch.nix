# Patched ExecuTorch v1.4.0 sources + static libs for Linux x86_64 and Android arm64.
{
  pkgs,
  pythonEnv,
  cmake,
  androidNdk,
  ndkApi,
}:
let
  inherit (pkgs) lib;

  version = "1.4.1";

  executorchSrc = pkgs.fetchFromGitHub {
    owner = "pytorch";
    repo = "executorch";
    rev = "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f"; # v1.4.0
    hash = "sha256-j43JX/3WLassyiyfwwIfSKO05ZmmS9wGUol4c+BZLEU=";
    fetchSubmodules = true;
  };

  cmakeLibs = [
    "libexecutorch.a"
    "libexecutorch_core.a"
    "extension/data_loader/libextension_data_loader.a"
    "extension/flat_tensor/libextension_flat_tensor.a"
    "extension/module/libextension_module_static.a"
    "extension/named_data_map/libextension_named_data_map.a"
    "extension/tensor/libextension_tensor.a"
    "extension/threadpool/libextension_threadpool.a"
    "kernels/portable/libportable_ops_lib.a"
    "kernels/portable/libportable_kernels.a"
    "backends/xnnpack/libxnnpack_backend.a"
    "backends/xnnpack/third-party/pthreadpool/libpthreadpool.a"
    "backends/xnnpack/third-party/cpuinfo/libcpuinfo.a"
    "backends/xnnpack/third-party/XNNPACK/libXNNPACK.a"
    "backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a"
    "backends/vulkan/libvulkan_backend.a"
  ];

  infersVulkanFfi = ./vulkan-ffi;
  infersVulkanFfiDir = "backends/vulkan/infers-ffi";

  src = pkgs.stdenv.mkDerivation {
    name = "executorch";
    inherit version;
    src = executorchSrc;
    patches = [
      ./patches/vulkan-gpu-input.patch
      ./patches/vulkan-external-adapter-teardown.patch
      ./patches/vulkan-sharedobject-algorithm.patch
      ./patches/vulkan-squeeze-algorithm.patch
    ];
    postPatch = ''
      mkdir -p ${infersVulkanFfiDir}
      cp ${infersVulkanFfi}/vulkan_external_adapter.cpp ${infersVulkanFfiDir}/
      cp ${infersVulkanFfi}/vulkan_gpu_input.cpp ${infersVulkanFfiDir}/
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p "$out"
      cp -r . "$out"
      runHook postInstall
    '';
    dontFixup = true;
  };

  commonCmakeFlags = [
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DPython3_EXECUTABLE=${pythonEnv}/bin/python"
    "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF"
    "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF"
    "-DEXECUTORCH_BUILD_PORTABLE_OPS=ON"
    "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON"
    "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON"
    "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON"
    "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON"
    "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON"
    "-DEXECUTORCH_BUILD_XNNPACK=ON"
    "-DEXECUTORCH_BUILD_VULKAN=ON"
    "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON"
    "-DEXECUTORCH_ENABLE_LOGGING=ON"
    "-DEXECUTORCH_BUILD_TESTS=OFF"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    "-DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF"
  ];

  buildLibs =
    {
      pname,
      extraCmakeFlags ? [ ],
      extraNativeBuildInputs ? [ ],
      extraPreConfigure ? "",
      ffiCxx ? "$CXX",
      ffiAr ? "$AR",
      ffiCxxFlags ? "-fPIC",
      extraAttrs ? { },
    }:
    pkgs.stdenv.mkDerivation (
      {
        inherit pname version src;
        nativeBuildInputs = [
          cmake
          pkgs.ninja
          pythonEnv
          pkgs.pkg-config
          pkgs.shaderc
        ]
        ++ extraNativeBuildInputs;
        buildInputs = [
          pkgs.vulkan-headers
          pkgs.vulkan-loader
          pkgs.zlib
        ];
        cmakeBuildType = "Release";
        cmakeFlags = commonCmakeFlags ++ extraCmakeFlags;
        preConfigure = ''
          export PATH="${pkgs.shaderc}/bin:$PATH"
          ${extraPreConfigure}
        '';
        enableParallelBuilding = true;
        dontStrip = true;
        installPhase = ''
          runHook preInstall
          cd "$NIX_BUILD_TOP/$sourceRoot/$cmakeBuildDir"
          ${lib.concatMapStrings (rel: ''
            if [ ! -f ${rel} ]; then
              echo "Missing required library: ${rel}" >&2
              exit 1
            fi
            install -Dm644 ${rel} "$out/${rel}"
          '') cmakeLibs}
          if [ -f kleidiai/libkleidiai.a ]; then
            install -Dm644 kleidiai/libkleidiai.a "$out/kleidiai/libkleidiai.a"
          fi

          et_root="$NIX_BUILD_TOP/$sourceRoot"
          vulkan_tp="$et_root/backends/vulkan/third-party"
          ffi_src="$et_root/${infersVulkanFfiDir}"
          ffi_build=$(mktemp -d)
          (
            ${ffiCxx} -std=c++17 -Wno-unused-parameter ${ffiCxxFlags} \
              -I"$et_root/src" \
              -I"$vulkan_tp/Vulkan-Headers/include" \
              -I"$vulkan_tp/volk" \
              -I"$vulkan_tp/VulkanMemoryAllocator" \
              -c "$ffi_src/vulkan_external_adapter.cpp" \
              -o "$ffi_build/vulkan_external_adapter.o"
            ${ffiCxx} -std=c++17 -Wno-unused-parameter ${ffiCxxFlags} \
              -I"$et_root/src" \
              -I"$vulkan_tp/Vulkan-Headers/include" \
              -I"$vulkan_tp/volk" \
              -I"$vulkan_tp/VulkanMemoryAllocator" \
              -c "$ffi_src/vulkan_gpu_input.cpp" \
              -o "$ffi_build/vulkan_gpu_input.o"
            ${ffiAr} rcs "$ffi_build/libinfers_et_vulkan_ffi.a" \
              "$ffi_build/vulkan_external_adapter.o" \
              "$ffi_build/vulkan_gpu_input.o"
            install -Dm644 "$ffi_build/libinfers_et_vulkan_ffi.a" "$out/backends/vulkan/libinfers_et_vulkan_ffi.a"
          )
          rm -rf "$ffi_build"
          runHook postInstall
        '';
      }
      // extraAttrs
    );

  linux = buildLibs {
    pname = "executorch-x86_64-unknown-linux-gnu";
  };

  android = buildLibs {
    pname = "executorch-android-arm64";
    extraCmakeFlags = [
      "-DCMAKE_TOOLCHAIN_FILE=${androidNdk}/build/cmake/android.toolchain.cmake"
      "-DANDROID_ABI=arm64-v8a"
      "-DANDROID_PLATFORM=android-${ndkApi}"
      "-DANDROID_NDK=${androidNdk}"
    ];
    extraPreConfigure = ''
      unset CC CXX
      export ANDROID_NDK_HOME=${androidNdk}
      export ANDROID_NDK_ROOT=${androidNdk}
    '';
    ffiCxx = "${androidNdk}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ndkApi}-clang++";
    ffiAr = "${androidNdk}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar";
    extraAttrs = {
      hardeningDisable = [ "all" ];
    };
  };

  libs = pkgs.runCommand "executorch-libs-${version}" { } ''
    mkdir -p "$out"
    ln -s ${linux} "$out/x86_64-unknown-linux-gnu"
    ln -s ${android} "$out/android-arm64"
  '';
in
{
  inherit
    src
    linux
    android
    libs
    version
    ;
}
