#![no_std]

use processing_core::{FitMode, ImageFormat, ProcessingOptions, Rotation};
use spirv_std::glam::{UVec3, Vec2};
use spirv_std::image::SampledImage;
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::{Image, spirv};

fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, tx: f32, ty: f32) -> f32 {
    let v0 = v00 + (v10 - v00) * tx;
    let v1 = v01 + (v11 - v01) * tx;
    v0 + (v1 - v0) * ty
}

fn yuv_to_rgb(y: f32, u: f32, v: f32) -> (f32, f32, f32) {
    let cb = u - 128.0;
    let cr = v - 128.0;
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    (r, g, b)
}

fn sample_u8_plane(src: &[u8], width: u32, height: u32, origin: usize, fx: f32, fy: f32) -> f32 {
    if fx < 0.0 || fy < 0.0 || fx >= width as f32 || fy >= height as f32 {
        return 0.0;
    }
    let x0 = fx.floor();
    let y0 = fy.floor();
    let x1 = (x0 + 1.0).min((width - 1) as f32);
    let y1 = (y0 + 1.0).min((height - 1) as f32);
    let tx = fx - x0;
    let ty = fy - y0;
    let w = width as usize;
    let i00 = origin + y0 as usize * w + x0 as usize;
    let i10 = origin + y0 as usize * w + x1 as usize;
    let i01 = origin + y1 as usize * w + x0 as usize;
    let i11 = origin + y1 as usize * w + x1 as usize;
    if i11 >= src.len() {
        return 0.0;
    }
    bilinear(
        src[i00] as f32,
        src[i10] as f32,
        src[i01] as f32,
        src[i11] as f32,
        tx,
        ty,
    )
}

fn sample_rgb888_bilinear(src: &[u8], src_w: u32, src_h: u32, fx: f32, fy: f32) -> (f32, f32, f32) {
    if fx < 0.0 || fy < 0.0 || fx >= src_w as f32 || fy >= src_h as f32 {
        return (0.0, 0.0, 0.0);
    }
    let x0 = fx.floor();
    let y0 = fy.floor();
    let x1 = (x0 + 1.0).min((src_w - 1) as f32);
    let y1 = (y0 + 1.0).min((src_h - 1) as f32);
    let tx = fx - x0;
    let ty = fy - y0;
    let w = src_w as usize;
    let i00 = (y0 as usize * w + x0 as usize) * 3;
    let i10 = (y0 as usize * w + x1 as usize) * 3;
    let i01 = (y1 as usize * w + x0 as usize) * 3;
    let i11 = (y1 as usize * w + x1 as usize) * 3;
    if i11 + 2 >= src.len() {
        return (0.0, 0.0, 0.0);
    }
    (
        bilinear(src[i00] as f32, src[i10] as f32, src[i01] as f32, src[i11] as f32, tx, ty),
        bilinear(
            src[i00 + 1] as f32,
            src[i10 + 1] as f32,
            src[i01 + 1] as f32,
            src[i11 + 1] as f32,
            tx,
            ty,
        ),
        bilinear(
            src[i00 + 2] as f32,
            src[i10 + 2] as f32,
            src[i01 + 2] as f32,
            src[i11 + 2] as f32,
            tx,
            ty,
        ),
    )
}

fn sample_nv12_chroma(src: &[u8], src_w: u32, src_h: u32, fx: f32, fy: f32) -> (f32, f32) {
    let cw = src_w / 2;
    let ch = src_h / 2;
    if cw == 0 || ch == 0 {
        return (128.0, 128.0);
    }
    let cx = fx * 0.5;
    let cy = fy * 0.5;
    if cx < 0.0 || cy < 0.0 || cx >= cw as f32 || cy >= ch as f32 {
        return (128.0, 128.0);
    }
    let x0 = cx.floor();
    let y0 = cy.floor();
    let x1 = (x0 + 1.0).min((cw - 1) as f32);
    let y1 = (y0 + 1.0).min((ch - 1) as f32);
    let tx = cx - x0;
    let ty = cy - y0;
    let origin = (src_w * src_h) as usize;
    let row = src_w as usize;
    let i00 = origin + y0 as usize * row + x0 as usize * 2;
    let i10 = origin + y0 as usize * row + x1 as usize * 2;
    let i01 = origin + y1 as usize * row + x0 as usize * 2;
    let i11 = origin + y1 as usize * row + x1 as usize * 2;
    if i11 + 1 >= src.len() {
        return (128.0, 128.0);
    }
    let u = bilinear(
        src[i00] as f32,
        src[i10] as f32,
        src[i01] as f32,
        src[i11] as f32,
        tx,
        ty,
    );
    let v = bilinear(
        src[i00 + 1] as f32,
        src[i10 + 1] as f32,
        src[i01 + 1] as f32,
        src[i11 + 1] as f32,
        tx,
        ty,
    );
    (u, v)
}

fn sample_nv12_bilinear(src: &[u8], src_w: u32, src_h: u32, fx: f32, fy: f32) -> (f32, f32, f32) {
    let y = sample_u8_plane(src, src_w, src_h, 0, fx, fy);
    let (u, v) = sample_nv12_chroma(src, src_w, src_h, fx, fy);
    yuv_to_rgb(y, u, v)
}

fn sample_i420_bilinear(src: &[u8], src_w: u32, src_h: u32, fx: f32, fy: f32) -> (f32, f32, f32) {
    let y_size = (src_w * src_h) as usize;
    let cw = src_w / 2;
    let ch = src_h / 2;
    let chroma = (cw * ch) as usize;
    let y = sample_u8_plane(src, src_w, src_h, 0, fx, fy);
    let u = sample_u8_plane(src, cw, ch, y_size, fx * 0.5, fy * 0.5);
    let v = sample_u8_plane(src, cw, ch, y_size + chroma, fx * 0.5, fy * 0.5);
    yuv_to_rgb(y, u, v)
}

fn sample_src(src: &[u8], params: &ProcessingOptions, fx: f32, fy: f32) -> (f32, f32, f32) {
    match params.src_format {
        ImageFormat::RGB888 => sample_rgb888_bilinear(src, params.src_w, params.src_h, fx, fy),
        ImageFormat::RGBF32 => (0.0, 0.0, 0.0),
        ImageFormat::NV12 => sample_nv12_bilinear(src, params.src_w, params.src_h, fx, fy),
        ImageFormat::I420 => sample_i420_bilinear(src, params.src_w, params.src_h, fx, fy),
    }
}

fn map_dst_to_src(params: &ProcessingOptions, dx: f32, dy: f32) -> (f32, f32, bool) {
    let (crop_x, crop_y, crop_w, crop_h) = {
        let (x, y, w, h) = params.effective_crop();
        (x as f32, y as f32, w as f32, h as f32)
    };

    let (rot_w, rot_h) = match params.rotation {
        Rotation::None | Rotation::R180DEG => (crop_w, crop_h),
        Rotation::R90DEG | Rotation::R270DEG => (crop_h, crop_w),
    };

    let dst_w = params.dest_w as f32;
    let dst_h = params.dest_h as f32;

    let (scale_x, scale_y, pad_x, pad_y) = match params.fit_mode {
        FitMode::STRETCH => (dst_w / rot_w, dst_h / rot_h, 0.0, 0.0),
        FitMode::CONTAIN => {
            let sx = dst_w / rot_w;
            let sy = dst_h / rot_h;
            let s = if sx < sy { sx } else { sy };
            let pad_x = (dst_w - rot_w * s) * 0.5;
            let pad_y = (dst_h - rot_h * s) * 0.5;
            (s, s, pad_x, pad_y)
        }
        FitMode::CROP => {
            let sx = dst_w / rot_w;
            let sy = dst_h / rot_h;
            let s = if sx > sy { sx } else { sy };
            let pad_x = (dst_w - rot_w * s) * 0.5;
            let pad_y = (dst_h - rot_h * s) * 0.5;
            (s, s, pad_x, pad_y)
        }
    };

    let rx = (dx - pad_x) / scale_x;
    let ry = (dy - pad_y) / scale_y;

    // GPU float pad can be a tiny positive, making the first dest row `ry < 0`.
    let eps = 0.001;
    if rx < -eps || ry < -eps || rx >= rot_w + eps || ry >= rot_h + eps {
        return (0.0, 0.0, false);
    }
    let rx = if rx < 0.0 { 0.0 } else { rx };
    let ry = if ry < 0.0 { 0.0 } else { ry };

    let (cx, cy) = match params.rotation {
        Rotation::None => (rx, ry),
        Rotation::R90DEG => (ry, crop_h - (rx + 1.0)),
        Rotation::R180DEG => (crop_w - (rx + 1.0), crop_h - (ry + 1.0)),
        Rotation::R270DEG => (crop_w - (ry + 1.0), rx),
    };

    // Clamp rather than reject: rust-gpu rotation can land 1 ulp past the last source row.
    let max_x = (params.src_w as f32 - 1.0).max(0.0);
    let max_y = (params.src_h as f32 - 1.0).max(0.0);
    let sx = (cx + crop_x).clamp(0.0, max_x);
    let sy = (cy + crop_y).clamp(0.0, max_y);

    (sx, sy, true)
}

fn write_rgb888(dst: &mut [u8], dst_w: u32, dx: u32, dy: u32, r: f32, g: f32, b: f32) {
    let o = ((dy * dst_w + dx) * 3) as usize;
    if o + 2 >= dst.len() {
        return;
    }
    dst[o] = r.round().clamp(0.0, 255.0) as u8;
    dst[o + 1] = g.round().clamp(0.0, 255.0) as u8;
    dst[o + 2] = b.round().clamp(0.0, 255.0) as u8;
}

fn write_rgbf32(dst: &mut [u8], dst_w: u32, dx: u32, dy: u32, r: f32, g: f32, b: f32) {
    let base = ((dy * dst_w + dx) * 12) as usize;
    if base + 11 >= dst.len() {
        return;
    }
    let r_bits = (r / 255.0).to_bits();
    dst[base] = (r_bits & 0xff) as u8;
    dst[base + 1] = ((r_bits >> 8) & 0xff) as u8;
    dst[base + 2] = ((r_bits >> 16) & 0xff) as u8;
    dst[base + 3] = ((r_bits >> 24) & 0xff) as u8;

    let g_bits = (g / 255.0).to_bits();
    dst[base + 4] = (g_bits & 0xff) as u8;
    dst[base + 5] = ((g_bits >> 8) & 0xff) as u8;
    dst[base + 6] = ((g_bits >> 16) & 0xff) as u8;
    dst[base + 7] = ((g_bits >> 24) & 0xff) as u8;

    let b_bits = (b / 255.0).to_bits();
    dst[base + 8] = (b_bits & 0xff) as u8;
    dst[base + 9] = ((b_bits >> 8) & 0xff) as u8;
    dst[base + 10] = ((b_bits >> 16) & 0xff) as u8;
    dst[base + 11] = ((b_bits >> 24) & 0xff) as u8;
}

fn write_dest_pixel(
    dst: &mut [u8],
    params: &ProcessingOptions,
    dx: u32,
    dy: u32,
    r: f32,
    g: f32,
    b: f32,
) {
    match params.dest_format {
        ImageFormat::RGB888 => write_rgb888(dst, params.dest_w, dx, dy, r, g, b),
        ImageFormat::RGBF32 => write_rgbf32(dst, params.dest_w, dx, dy, r, g, b),
        ImageFormat::NV12 | ImageFormat::I420 => {}
    }
}

/// Storage-buffer convert path. `params.src_format` selects RGB888, RGBF32, NV12, or I420.
#[spirv(compute(threads(16, 16)))]
pub fn convert_main(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] params: &ProcessingOptions,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] src: &[u8],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] dst: &mut [u8],
) {
    let dx = gid.x;
    let dy = gid.y;
    if dx >= params.dest_w || dy >= params.dest_h {
        return;
    }

    let (sx, sy, valid) = map_dst_to_src(params, dx as f32, dy as f32);
    let (r, g, b) = if valid {
        sample_src(src, params, sx, sy)
    } else {
        (0.0, 0.0, 0.0)
    };

    write_dest_pixel(dst, params, dx, dy, r, g, b);
}

/// Sampled-image convert path.
///
/// The sampled type is `f32` and format-agnostic: RGB UNORM views and YUV images
/// bound with `VkSamplerYcbcrConversion` both arrive as RGB in 0..1.
#[spirv(compute(threads(16, 16)))]
pub fn convert_image(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] params: &ProcessingOptions,
    #[spirv(descriptor_set = 0, binding = 1)] src: &SampledImage<Image!(2D, type = f32, sampled)>,
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] dst: &mut [u8],
) {
    let dx = gid.x;
    let dy = gid.y;
    if dx >= params.dest_w || dy >= params.dest_h {
        return;
    }

    let (sx, sy, valid) = map_dst_to_src(params, dx as f32, dy as f32);
    let (r, g, b) = if valid {
        let u = (sx + 0.5) / params.src_w as f32;
        let v = (sy + 0.5) / params.src_h as f32;
        let sampled = src.sample_by_lod(Vec2::new(u, v), 0.0);
        (
            sampled.x * 255.0,
            sampled.y * 255.0,
            sampled.z * 255.0,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    write_dest_pixel(dst, params, dx, dy, r, g, b);
}
