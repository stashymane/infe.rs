use std::path::{Path, PathBuf};

use infers_core::{CoreError, Cpu, DataType, Tensor};

use crate::{cpu_tensor_f32, read_f32_output};

/// Locate `target/yolo26n-face` by walking up from this crate's manifest dir.
pub fn yolo26n_face_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("INFERS_YOLO26N_FACE_DIR") {
        let path = PathBuf::from(dir);
        if path.join("manifest.yaml").is_file() {
            return Some(path);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join("target/yolo26n-face");
        if candidate.join("manifest.yaml").is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn yolo26n_face_asset(subpath: &str) -> Option<PathBuf> {
    let path = yolo26n_face_dir()?.join(subpath);
    path.exists().then_some(path)
}

pub fn yolo26n_face_imgsz() -> Option<u32> {
    let manifest = yolo26n_face_dir()?.join("manifest.yaml");
    let content = std::fs::read_to_string(manifest).ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("imgsz:") {
            return line.split(':').nth(1)?.trim().parse().ok();
        }
    }
    None
}

pub fn read_model_bytes(path: &Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Assert that inference produced the expected number of outputs with matching shapes and finite f32 values.
pub fn assert_detector_inference_outputs(
    outputs: &[Tensor<Cpu>],
    expected_shapes: &[infers_core::TensorShape],
) {
    assert_eq!(
        outputs.len(),
        expected_shapes.len(),
        "output count mismatch"
    );
    for (output, expected_shape) in outputs.iter().zip(expected_shapes) {
        assert_eq!(output.shape(), expected_shape);
        assert_eq!(output.dtype(), DataType::F32);
        let values = read_f32_output(std::slice::from_ref(output)).expect("read f32 output");
        assert!(
            !values.is_empty(),
            "output tensor {:?} is empty",
            expected_shape.dims()
        );
        assert!(
            values.iter().all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }
}

/// Zero-filled NCHW f32 tensor sized for the yolo26n-face detector input.
pub fn detector_input_zeros(imgsz: u32) -> Result<infers_core::Tensor<infers_core::Cpu>, CoreError> {
    let shape = infers_core::TensorShape::new([1, 3, imgsz as usize, imgsz as usize])?;
    let count = shape.element_count();
    cpu_tensor_f32(shape, &vec![0.0f32; count])
}
