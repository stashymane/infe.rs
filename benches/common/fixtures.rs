use infers::{HardwareImage, ImageFormat, ProcessingOptions, Rotation, TensorLayout, TensorShape};
use processing::FitMode;

pub const FRAME_W: u32 = 1280;
pub const FRAME_H: u32 = 720;

pub fn camera_frame(width: u32, height: u32) -> HardwareImage {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 255) / width.max(1)) as u8);
            data.push(((y * 255) / height.max(1)) as u8);
            data.push((((x + y) * 255) / (width + height).max(1)) as u8);
        }
    }
    HardwareImage::new(width, height, ImageFormat::Rgb888, data).expect("synthetic camera frame")
}

pub fn detector_options(src_w: u32, src_h: u32, imgsz: u32) -> ProcessingOptions {
    let dest_format = ImageFormat::Rgbf32;
    ProcessingOptions {
        src_w,
        src_h,
        crop_x: 0,
        crop_y: 0,
        crop_w: src_w,
        crop_h: src_h,
        dest_w: imgsz,
        dest_h: imgsz,
        src_format: ImageFormat::Rgb888,
        dest_format,
        fit_mode: FitMode::Contain,
        rotation: Rotation::None,
        dest_layout: TensorLayout::default_for_dest_format(dest_format),
    }
}

pub fn imgsz_from_input_shape(shape: &TensorShape) -> u32 {
    let dims = shape.dims();
    assert_eq!(
        dims.len(),
        4,
        "expected NCHW input shape [1, 3, H, W], got {dims:?}"
    );
    assert_eq!(dims[0], 1, "batch must be 1");
    assert_eq!(dims[1], 3, "channels must be 3 (NCHW)");
    assert_eq!(dims[2], dims[3], "square model input expected");
    dims[2] as u32
}
