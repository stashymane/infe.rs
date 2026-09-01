use infers_bindings::*;

mod common;

fn bindings_processing_options(
    opts: infers_core::ProcessingOptions,
) -> ProcessingOptions {
    ProcessingOptions {
        src_w: opts.src_w,
        src_h: opts.src_h,
        crop_x: opts.crop_x,
        crop_y: opts.crop_y,
        crop_w: opts.crop_w,
        crop_h: opts.crop_h,
        dest_w: opts.dest_w,
        dest_h: opts.dest_h,
        src_format: opts.src_format.into(),
        dest_format: opts.dest_format.into(),
        fit_mode: opts.fit_mode.into(),
        rotation: opts.rotation.into(),
        dest_layout: opts.dest_layout.into(),
    }
}

#[test]
fn test_cpu_preprocess_pipeline() {
    let processor = CpuImageProcessor::new();
    let frame = infers_test_utils::camera_frame_640x480(128);
    let host = create_host_image(
        frame.width(),
        frame.height(),
        ImageFormat::Rgb888,
        frame.as_bytes().to_vec(),
    )
    .expect("host image");

    let options = bindings_processing_options(infers_test_utils::detector_preprocess_options(224));
    let tensor = processor.process(host, options).expect("preprocess");
    assert_eq!(
        tensor.shape(),
        TensorShape {
            dims: vec![1, 3, 224, 224]
        }
    );
    assert_eq!(tensor.dtype(), DataType::F32);
    assert_eq!(
        tensor.read_bytes().expect("read bytes").len(),
        1 * 3 * 224 * 224 * 4
    );
}
