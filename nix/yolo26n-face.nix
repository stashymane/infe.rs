# YOLO26n-face → ExecuTorch .pte export (XNNPACK + Vulkan).
{
  pkgs,
  executorchExport,
}:
let
  inherit (pkgs) lib;

  manifestYaml = ../assets/yolo26n-face-manifest.yaml;
  manifestText = builtins.readFile manifestYaml;

  readManifestField = section: key:
    let
      hits = lib.filter (
        line:
        lib.hasInfix "${key}:" line
        && !(lib.hasPrefix "#" (lib.trim line))
      ) (lib.splitString "\n" section);
    in
    if hits == [ ] then
      null
    else
      lib.trim (lib.elemAt (lib.splitString "${key}:" (lib.head hits)) 1);

  weightsSection =
    let
      afterWeights = lib.elemAt (lib.splitString "weights:" manifestText) 1;
    in
    lib.head (lib.splitString "\nartifacts:" afterWeights);

  weightsSpec = {
    url = readManifestField weightsSection "url";
    filename = readManifestField weightsSection "filename";
    nix_hash = readManifestField weightsSection "nix_hash";
  };

  exportSrc = pkgs.linkFarm "yolo26n-face-export-src" [
    { name = "manifest.yaml"; path = manifestYaml; }
    { name = "yolo26n_face_export.py"; path = ./export/yolo26n_face_export.py; }
    { name = "executorch_export.py"; path = ./export/executorch_export.py; }
  ];

  weights = pkgs.fetchurl {
    name = weightsSpec.filename or "yolo26_widerdataset.pt";
    url = weightsSpec.url;
    hash = weightsSpec.nix_hash;
  };

  asset = pkgs.stdenv.mkDerivation {
    pname = "yolo26n-face-export";
    version = builtins.substring 0 12 (builtins.hashFile "sha256" manifestYaml);
    src = exportSrc;
    nativeBuildInputs = [ executorchExport.exportPython ];
    inherit (executorchExport.exportEnv)
      LD_LIBRARY_PATH
      MPLBACKEND
      YOLO_VERBOSE
      HF_HUB_OFFLINE
      TRANSFORMERS_OFFLINE
      ;
    SMOKE_TEST = "1";
    buildPhase = ''
      runHook preBuild
      export HOME="$TMPDIR"
      export OUTPUT_DIR="$out"
      export PYTHONPATH="$src"

      work="$TMPDIR/yolo26n-face"
      mkdir -p "$work"
      manifest="$work/manifest.yaml"
      export MANIFEST_SRC="$src/manifest.yaml"
      export MANIFEST_EDIT="$manifest"
      export WEIGHTS_PATH="${weights}"
      ${executorchExport.exportPython}/bin/python -c '
import os
import yaml
from pathlib import Path

data = yaml.safe_load(Path(os.environ["MANIFEST_SRC"]).read_text())
data.setdefault("weights", {})["path"] = os.environ["WEIGHTS_PATH"]
Path(os.environ["MANIFEST_EDIT"]).write_text(yaml.safe_dump(data, sort_keys=False))
'
      export MANIFEST_PATH="$manifest"
      mkdir -p "$out"
      ${executorchExport.exportPython}/bin/python "$src/yolo26n_face_export.py"
      runHook postBuild
    '';
    installPhase = "true";
    dontFixup = true;
  };
in
{
  inherit asset;
}
