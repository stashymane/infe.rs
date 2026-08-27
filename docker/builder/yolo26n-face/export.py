#!/usr/bin/env python3
"""Export YOLO26n-face to ExecuTorch .pte files for XNNPACK and Vulkan."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import YAML as UltralyticsYAML

MANIFEST_NAME = "manifest.yaml"
GENERATED_MANIFEST_NAME = "manifest.generated.yaml"



def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(url: str, dest: Path, expected_sha256: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights from {url} ...")
    urllib.request.urlretrieve(url, dest)
    actual = sha256_file(dest)
    if expected_sha256 and actual != expected_sha256.lower():
        raise RuntimeError(
            f"Weight SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    print(f"Weights saved to {dest} ({dest.stat().st_size} bytes, sha256={actual})")


def resolve_weights(manifest: dict[str, Any], output_dir: Path) -> Path:
    override = os.environ.get("YOLO26N_FACE_WEIGHTS")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO26N_FACE_WEIGHTS not found: {path}")
        print(f"Using YOLO26N_FACE_WEIGHTS={path}")
        return path

    weights = manifest.get("weights") or {}
    url = weights.get("url")
    if not url:
        raise RuntimeError(
            "No weights configured. Set weights.url in manifest.yaml or YOLO26N_FACE_WEIGHTS."
        )

    filename = weights.get("filename") or Path(url).name or "yolo26n-face.pt"
    dest = output_dir / "weights" / filename
    if dest.is_file():
        actual = sha256_file(dest)
        expected = (weights.get("sha256") or "").lower()
        if not expected or actual == expected:
            print(f"Reusing cached weights at {dest}")
            return dest

    download_weights(url, dest, weights.get("sha256"))
    return dest


def torch2executorch_vulkan(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    from executorch import version as executorch_version
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.exir import to_edge_transform_and_lower

    print(f"{prefix} starting Vulkan export with ExecuTorch {executorch_version.__version__}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pte_file = output_dir / "model.pte"
    et_program = to_edge_transform_and_lower(
        torch.export.export(model, (im,)),
        partitioner=[VulkanPartitioner({"force_fp16": True})],
    ).to_executorch()
    pte_file.write_bytes(et_program.buffer)

    if metadata is not None:
        UltralyticsYAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)


def export_backends(weights_path: Path, imgsz: int, work_dir: Path) -> tuple[Path, Path]:
    import ultralytics.utils.export.executorch as et_export

    local_weights = work_dir / weights_path.name
    shutil.copy2(weights_path, local_weights)

    yolo = YOLO(str(local_weights))

    original_torch2executorch = et_export.torch2executorch
    vulkan_root: Path | None = None
    xnnpack_root: Path | None = None

    vulkan_out = work_dir / "vulkan"
    xnnpack_out = work_dir / "xnnpack"
    vulkan_out.mkdir(parents=True, exist_ok=True)
    xnnpack_out.mkdir(parents=True, exist_ok=True)

    cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        et_export.torch2executorch = torch2executorch_vulkan
        vulkan_root = Path(
            yolo.export(format="executorch", imgsz=imgsz, batch=1, device="cpu")
        ).resolve()
        vulkan_pte_src = vulkan_root / "model.pte"
        if not vulkan_pte_src.is_file():
            raise RuntimeError(f"Vulkan export missing {vulkan_pte_src}")
        shutil.copy2(vulkan_pte_src, vulkan_out / "model.pte")
        vulkan_meta = vulkan_root / "metadata.yaml"
        if vulkan_meta.is_file():
            shutil.copy2(vulkan_meta, vulkan_out / "metadata.yaml")

        et_export.torch2executorch = original_torch2executorch
        xnnpack_root = Path(
            yolo.export(format="executorch", imgsz=imgsz, batch=1, device="cpu")
        ).resolve()
        xnnpack_pte_src = xnnpack_root / "model.pte"
        if not xnnpack_pte_src.is_file():
            raise RuntimeError(f"XNNPACK export missing {xnnpack_pte_src}")
        shutil.copy2(xnnpack_pte_src, xnnpack_out / "model.pte")
        xnnpack_meta = xnnpack_root / "metadata.yaml"
        if xnnpack_meta.is_file():
            shutil.copy2(xnnpack_meta, xnnpack_out / "metadata.yaml")
    finally:
        et_export.torch2executorch = original_torch2executorch
        os.chdir(cwd)

    for root in (vulkan_root, xnnpack_root):
        if root is not None and root.is_dir() and root.parent.resolve() == work_dir.resolve():
            shutil.rmtree(root, ignore_errors=True)

    return xnnpack_out / "model.pte", vulkan_out / "model.pte"


def smoke_test_pte(pte_path: Path, imgsz: int) -> None:
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
    except ImportError:
        print("Skipping smoke test: executorch pybindings unavailable")
        return

    program = _load_for_executorch(str(pte_path))
    method = program.load_method("forward")
    example = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
    outputs = method.execute([example])
    if not outputs:
        raise RuntimeError(f"Smoke test produced no outputs for {pte_path}")
    print(f"Smoke test OK: {pte_path.name} -> {len(outputs)} output(s)")


def write_manifest(
    generated_path: Path,
    manifest: dict[str, Any],
    imgsz: int,
    xnnpack_pte: Path,
    vulkan_pte: Path,
    weights_path: Path,
) -> None:
    import ultralytics

    try:
        from executorch import version as executorch_version

        et_version = executorch_version.__version__
    except Exception:
        et_version = "unknown"

    manifest["imgsz"] = imgsz
    manifest["weights"] = manifest.get("weights") or {}
    manifest["weights"]["resolved_path"] = str(weights_path)
    manifest["weights"]["sha256"] = sha256_file(weights_path)
    # Preserve template fields if present.
    manifest["weights"].setdefault("url", None)
    manifest["weights"].setdefault("filename", weights_path.name)
    manifest["weights"].setdefault("source", "override")

    artifacts = manifest.setdefault("artifacts", {})
    xnnpack = artifacts.setdefault("xnnpack", {})
    vulkan = artifacts.setdefault("vulkan", {})
    xnnpack["path"] = "xnnpack/model.pte"
    xnnpack["sha256"] = sha256_file(xnnpack_pte)
    xnnpack["bytes"] = xnnpack_pte.stat().st_size
    vulkan["path"] = "vulkan/model.pte"
    vulkan["sha256"] = sha256_file(vulkan_pte)
    vulkan["bytes"] = vulkan_pte.stat().st_size

    manifest["export"] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),
        "executorch_version": str(et_version),
        "ultralytics_version": str(ultralytics.__version__),
        "input_shape": [1, 3, imgsz, imgsz],
        "input_layout": "NCHW",
        "input_dtype": "float32",
    }

    # Write beside the committed template so local builds do not dirty git.
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = generated_path.with_suffix(".yaml.tmp")
    with tmp_manifest.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    tmp_manifest.replace(generated_path)


def main() -> int:
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output")).resolve()
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"Missing manifest template at {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    imgsz = int(os.environ.get("IMGSZ", manifest.get("imgsz", 192)))
    if imgsz % 32 != 0:
        print(f"IMGSZ must be a multiple of 32, got {imgsz}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = resolve_weights(manifest, output_dir)

    with tempfile.TemporaryDirectory(prefix="yolo26n-face-export-") as tmp:
        work_dir = Path(tmp)
        print(f"Exporting yolo26n-face at imgsz={imgsz} ...")
        xnnpack_pte, vulkan_pte = export_backends(weights_path, imgsz, work_dir)

        if os.environ.get("SMOKE_TEST", "1") != "0":
            smoke_test_pte(xnnpack_pte, imgsz)
            print("Skipping Vulkan smoke test (no GPU in export container).")

        xnnpack_out = output_dir / "xnnpack"
        vulkan_out = output_dir / "vulkan"
        xnnpack_out.mkdir(parents=True, exist_ok=True)
        vulkan_out.mkdir(parents=True, exist_ok=True)

        shutil.copy2(xnnpack_pte, xnnpack_out / "model.pte")
        shutil.copy2(vulkan_pte, vulkan_out / "model.pte")
        xnnpack_meta = work_dir / "xnnpack" / "metadata.yaml"
        vulkan_meta = work_dir / "vulkan" / "metadata.yaml"
        if xnnpack_meta.is_file():
            shutil.copy2(xnnpack_meta, xnnpack_out / "metadata.yaml")
        if vulkan_meta.is_file():
            shutil.copy2(vulkan_meta, vulkan_out / "metadata.yaml")

    write_manifest(
        output_dir / GENERATED_MANIFEST_NAME,
        manifest,
        imgsz,
        output_dir / "xnnpack" / "model.pte",
        output_dir / "vulkan" / "model.pte",
        weights_path,
    )
    print(f"Export complete -> {output_dir}")
    print(f"Generated manifest -> {output_dir / GENERATED_MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
