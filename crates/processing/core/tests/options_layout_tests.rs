use processing_core::{FitMode, ImageFormat, ProcessingOptions, map_dst_to_src};

#[test]
fn processing_options_matches_shader_push_constant_layout() {
    assert_eq!(core::mem::size_of::<ProcessingOptions>(), 52);
}

#[test]
fn map_dst_to_src_identity_and_cardinal_angles() {
    let base = ProcessingOptions {
        src_w: 40,
        src_h: 20,
        crop_w: 40,
        crop_h: 20,
        dest_w: 40,
        dest_h: 20,
        src_format: ImageFormat::Rgb888,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Stretch,
        rotation_degrees: 0.0,
        ..Default::default()
    };

    let (sx, sy, valid) = map_dst_to_src(&base, 10.0, 5.0);
    assert!(valid);
    assert!((sx - 10.0).abs() < 1e-4 && (sy - 5.0).abs() < 1e-4);

    let rot90 = ProcessingOptions {
        dest_w: 20,
        dest_h: 40,
        rotation_degrees: 90.0,
        ..base
    };
    // Dest top-left maps near crop bottom-left under clockwise rotation.
    let (sx, sy, valid) = map_dst_to_src(&rot90, 0.0, 0.0);
    assert!(valid);
    assert!(sx < 1.0, "sx={sx}");
    assert!((sy - 19.0).abs() < 1.5, "sy={sy}");
}

#[test]
fn map_dst_to_src_oblique_uses_aabb_letterbox() {
    let opts = ProcessingOptions {
        src_w: 40,
        src_h: 20,
        crop_w: 40,
        crop_h: 20,
        dest_w: 80,
        dest_h: 80,
        src_format: ImageFormat::Rgb888,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Contain,
        rotation_degrees: 45.0,
        ..Default::default()
    };
    // Extreme corner of a Contain canvas is letterbox for a 45° AABB fit.
    let (_sx, _sy, valid) = map_dst_to_src(&opts, 0.0, 0.0);
    assert!(!valid);
    let (_sx, _sy, valid) = map_dst_to_src(&opts, 40.0, 40.0);
    assert!(valid);
}
