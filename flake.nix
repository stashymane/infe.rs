{
  description = "Native toolchains and ExecuTorch libraries for infers";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          android_sdk.accept_license = true;
        };
      };
      deps = import ./nix/deps.nix { inherit pkgs; };
      executorch = import ./nix/executorch.nix {
        inherit pkgs;
        inherit (deps)
          pythonEnv
          cmake
          androidNdk
          ndkApi
          ;
      };
      executorchExport = import ./nix/executorch-export.nix {
        inherit pkgs;
        inherit (deps) python torch;
      };
      yolo26nFace = import ./nix/yolo26n-face.nix {
        inherit pkgs executorchExport;
      };
    in
    {
      packages.${system} = {
        default = executorch.libs;
        deps-env = deps.depsEnv;
        android-ndk = deps.androidNdkPkg;
        executorch-src = executorch.src;
        executorch-x86_64-unknown-linux-gnu = executorch.linux;
        executorch-android-arm64 = executorch.android;
        executorch-libs = executorch.libs;
        yolo26n-face-export = yolo26nFace.asset;
        yolo26n-face = yolo26nFace.asset;
      };

      apps.${system} = {
        deps-env = {
          type = "app";
          program = "${deps.depsEnv}/bin/deps-env";
        };
      };

      devShells.${system}.default = deps.depsShell;
    };
}
