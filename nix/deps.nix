# Native toolchain for ExecuTorch cmake (host x86_64 Linux + Android NDK).
{
  pkgs,
}:
let
  inherit (pkgs) lib;

  ndkApi = "26";
  torchVersion = "2.13.0";

  cmake = pkgs.cmake_3 or pkgs.cmake;

  python = pkgs.python312;

  # nixpkgs androidenv (NDK 29.0.14206865); requires allowUnfree + android_sdk.accept_license.
  androidNdkPkg = pkgs.androidenv.androidPkgs.ndk-bundle;
  androidPlatformTools = pkgs.androidenv.androidPkgs.platform-tools;
  androidNdk = "${androidNdkPkg}/libexec/android-sdk/ndk-bundle";
  ndkVersion = androidNdkPkg.version;

  # Official CPU wheel (same pin as the old pip install). nixpkgs torch is not 2.13.
  torch = python.pkgs.buildPythonPackage {
    pname = "torch";
    version = "${torchVersion}+cpu";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://download.pytorch.org/whl/cpu/torch-${torchVersion}%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl";
      hash = "sha256-TKSpOUsMdxI4pPc1kP27xN662F7Q+mPQJq4bCF2n1uI=";
    };
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.llvmPackages.openmp
    ];
    autoPatchelfIgnoreMissingDeps = true;
    propagatedBuildInputs = with python.pkgs; [
      filelock
      fsspec
      jinja2
      networkx
      setuptools
      sympy
      typing-extensions
    ];
    pythonImportsCheck = [ ];
    doCheck = false;
  };

  pythonEnv = python.withPackages (ps: [
    ps.packaging
    ps.pyyaml
    torch
  ]);

  toolchainPkgs =
    [
      androidNdkPkg
      androidPlatformTools
      cmake
      pythonEnv
    ]
    ++ (with pkgs; [
      bash
      binutils
      nix
      cacert
      coreutils
      findutils
      gawk
      gcc
      git
      gnugrep
      gnumake
      gnused
      gnutar
      gzip
      ninja
      patch
      pkg-config
      shaderc
      stdenv.cc.cc.lib
      unzip
      vulkan-headers
      vulkan-loader
      which
      zlib
    ]);

  envProfile = ''
    export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
    export GIT_SSL_CAINFO="$SSL_CERT_FILE"
    export NIX_SSL_CERT_FILE="$SSL_CERT_FILE"
    export ANDROID_NDK_HOME=${androidNdk}
    export ANDROID_NDK_ROOT="$ANDROID_NDK_HOME"
    export ANDROID_NDK_API=${ndkApi}
    export PYTHON3=${pythonEnv}/bin/python3
    # Host glslc (shaderc) must precede the NDK copy, which lacks GL_EXT_integer_dot_product.
    export PATH=${lib.makeBinPath [
      androidPlatformTools
      pkgs.shaderc
      pythonEnv
      cmake
      pkgs.nix
      pkgs.ninja
      pkgs.git
      pkgs.gcc
      pkgs.gnumake
      pkgs.coreutils
      pkgs.findutils
      pkgs.gnugrep
      pkgs.gnused
      pkgs.gawk
      pkgs.gnutar
      pkgs.gzip
      pkgs.which
      pkgs.bash
    ]}:"$PATH"
  '';

  depsEnv = pkgs.buildFHSEnv {
    pname = "deps-env";
    version = "1";
    targetPkgs = _: toolchainPkgs;
    extraOutputsToInstall = [ "dev" ];
    profile = envProfile;
    runScript = "bash";
  };

  depsShell = pkgs.mkShell {
    name = "infers-deps";
    packages = toolchainPkgs;
    ANDROID_NDK_HOME = androidNdk;
    ANDROID_NDK_ROOT = androidNdk;
    ANDROID_NDK_API = ndkApi;
    PYTHON3 = "${pythonEnv}/bin/python3";
    SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    GIT_SSL_CAINFO = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    shellHook = envProfile;
  };
in
{
  inherit
    androidNdk
    androidNdkPkg
    cmake
    depsEnv
    depsShell
    ndkApi
    ndkVersion
    pythonEnv
    python
    torch
    toolchainPkgs
    ;
}
