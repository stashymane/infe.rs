//! Dest-centric image convert shared by CPU and GPU (storage-buffer) paths.

use crate::{FitMode, ImageFormat, ProcessingOptions, Rotation, TensorLayout};

#[inline(always)]
fn floor_f32(x: f32) -> f32 {
    let mut i = x as i32;
    if (i as f32) > x {
        i -= 1;
    }
    i as f32
}

#[inline(always)]
fn round_f32(x: f32) -> f32 {
    floor_f32(x + 0.5)
}

#[inline(always)]
fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, tx: f32, ty: f32) -> f32 {
    let v0 = v00 + (v10 - v00) * tx;
    let v1 = v01 + (v11 - v01) * tx;
    v0 + (v1 - v0) * ty
}

#[inline(always)]
fn yuv_to_rgb(y: f32, u: f32, v: f32) -> (f32, f32, f32) {
    let cb = u - 128.0;
    let cr = v - 128.0;
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    (r, g, b)
}

#[inline(always)]
fn sample_u8_plane(src: &[u8], width: u32, height: u32, origin: usize, fx: f32, fy: f32) -> f32 {
    if fx < 0.0 || fy < 0.0 || fx >= width as f32 || fy >= height as f32 {
        return 0.0;
    }
    let x0 = floor_f32(fx);
    let y0 = floor_f32(fy);
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

#[inline(always)]
fn sample_rgb888_bilinear(src: &[u8], src_w: u32, src_h: u32, fx: f32, fy: f32) -> (f32, f32, f32) {
    if fx < 0.0 || fy < 0.0 || fx >= src_w as f32 || fy >= src_h as f32 {
        return (0.0, 0.0, 0.0);
    }
    let x0 = floor_f32(fx);
    let y0 = floor_f32(fy);
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

#[inline(always)]
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
    let x0 = floor_f32(cx);
    let y0 = floor_f32(cy);
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

#[inline(always)]
fn sample_nv12_bilinear(src: &[u8], src_w: u32, src_h: u32, fx: f32, fy: f32) -> (f32, f32, f32) {
    let y = sample_u8_plane(src, src_w, src_h, 0, fx, fy);
    let (u, v) = sample_nv12_chroma(src, src_w, src_h, fx, fy);
    yuv_to_rgb(y, u, v)
}

#[inline(always)]
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

/// Sample source bytes at floating source coordinates (pre-rotation crop space mapped).
#[inline(always)]
pub fn sample_src(src: &[u8], params: &ProcessingOptions, fx: f32, fy: f32) -> (f32, f32, f32) {
    match params.src_format {
        ImageFormat::Rgb888 => sample_rgb888_bilinear(src, params.src_w, params.src_h, fx, fy),
        ImageFormat::Rgbf32 => (0.0, 0.0, 0.0),
        ImageFormat::Nv12 => sample_nv12_bilinear(src, params.src_w, params.src_h, fx, fy),
        ImageFormat::I420 => sample_i420_bilinear(src, params.src_w, params.src_h, fx, fy),
    }
}

/// Map a destination pixel center to source coordinates. `valid` is false in letterbox padding.
#[inline(always)]
pub fn map_dst_to_src(params: &ProcessingOptions, dx: f32, dy: f32) -> (f32, f32, bool) {
    let (crop_x, crop_y, crop_w, crop_h) = {
        let (x, y, w, h) = params.effective_crop();
        (x as f32, y as f32, w as f32, h as f32)
    };

    let (rot_w, rot_h) = match params.rotation {
        Rotation::None | Rotation::Rot180 => (crop_w, crop_h),
        Rotation::Rot90 | Rotation::Rot270 => (crop_h, crop_w),
    };

    let dst_w = params.dest_w as f32;
    let dst_h = params.dest_h as f32;

    let (scale_x, scale_y, pad_x, pad_y) = match params.fit_mode {
        FitMode::Stretch => (dst_w / rot_w, dst_h / rot_h, 0.0, 0.0),
        FitMode::Contain => {
            let sx = dst_w / rot_w;
            let sy = dst_h / rot_h;
            let s = if sx < sy { sx } else { sy };
            let pad_x = (dst_w - rot_w * s) * 0.5;
            let pad_y = (dst_h - rot_h * s) * 0.5;
            (s, s, pad_x, pad_y)
        }
        FitMode::Crop => {
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
        Rotation::Rot90 => (ry, crop_h - (rx + 1.0)),
        Rotation::Rot180 => (crop_w - (rx + 1.0), crop_h - (ry + 1.0)),
        Rotation::Rot270 => (crop_w - (ry + 1.0), rx),
    };

    // Clamp rather than reject: rust-gpu rotation can land 1 ulp past the last source row.
    let max_x = (params.src_w as f32 - 1.0).max(0.0);
    let max_y = (params.src_h as f32 - 1.0).max(0.0);
    let sx = (cx + crop_x).clamp(0.0, max_x);
    let sy = (cy + crop_y).clamp(0.0, max_y);

    (sx, sy, true)
}

#[inline(always)]
fn write_rgb888(dst: &mut [u8], dst_w: u32, dx: u32, dy: u32, r: f32, g: f32, b: f32) {
    let o = ((dy * dst_w + dx) * 3) as usize;
    if o + 2 >= dst.len() {
        return;
    }
    dst[o] = round_f32(r).clamp(0.0, 255.0) as u8;
    dst[o + 1] = round_f32(g).clamp(0.0, 255.0) as u8;
    dst[o + 2] = round_f32(b).clamp(0.0, 255.0) as u8;
}

#[inline(always)]
fn write_rgb888_nchw(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    dx: u32,
    dy: u32,
    r: f32,
    g: f32,
    b: f32,
) {
    let hw = (dst_w * dst_h) as usize;
    let i = (dy * dst_w + dx) as usize;
    if i >= hw {
        return;
    }
    dst[i] = round_f32(r).clamp(0.0, 255.0) as u8;
    dst[hw + i] = round_f32(g).clamp(0.0, 255.0) as u8;
    dst[2 * hw + i] = round_f32(b).clamp(0.0, 255.0) as u8;
}

#[inline(always)]
pub fn store_f32_le(dst: &mut [u8], offset: usize, value: f32) {
    if offset + 4 > dst.len() {
        return;
    }
    let bits = value.to_bits();
    dst[offset] = (bits & 0xff) as u8;
    dst[offset + 1] = ((bits >> 8) & 0xff) as u8;
    dst[offset + 2] = ((bits >> 16) & 0xff) as u8;
    dst[offset + 3] = ((bits >> 24) & 0xff) as u8;
}

#[inline(always)]
fn write_rgbf32(dst: &mut [u8], dst_w: u32, dx: u32, dy: u32, r: f32, g: f32, b: f32) {
    let base = ((dy * dst_w + dx) * 12) as usize;
    store_f32_le(dst, base, r / 255.0);
    store_f32_le(dst, base + 4, g / 255.0);
    store_f32_le(dst, base + 8, b / 255.0);
}

#[inline(always)]
fn write_rgbf32_nchw(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    dx: u32,
    dy: u32,
    r: f32,
    g: f32,
    b: f32,
) {
    let hw = (dst_w * dst_h) as usize;
    let i = (dy * dst_w + dx) as usize;
    if i >= hw {
        return;
    }
    store_f32_le(dst, i * 4, r / 255.0);
    store_f32_le(dst, (hw + i) * 4, g / 255.0);
    store_f32_le(dst, (2 * hw + i) * 4, b / 255.0);
}

/// Write one destination pixel in the configured format/layout. RGB channels are 0..255.
#[inline(always)]
pub fn write_dest_pixel(
    dst: &mut [u8],
    params: &ProcessingOptions,
    dx: u32,
    dy: u32,
    r: f32,
    g: f32,
    b: f32,
) {
    match (params.dest_format, params.dest_layout) {
        (ImageFormat::Rgb888, TensorLayout::Nhwc) => {
            write_rgb888(dst, params.dest_w, dx, dy, r, g, b)
        }
        (ImageFormat::Rgb888, TensorLayout::Nchw) => {
            write_rgb888_nchw(dst, params.dest_w, params.dest_h, dx, dy, r, g, b)
        }
        (ImageFormat::Rgbf32, TensorLayout::Nhwc) => {
            write_rgbf32(dst, params.dest_w, dx, dy, r, g, b)
        }
        (ImageFormat::Rgbf32, TensorLayout::Nchw) => {
            write_rgbf32_nchw(dst, params.dest_w, params.dest_h, dx, dy, r, g, b)
        }
        (ImageFormat::Nv12 | ImageFormat::I420, _) => {}
    }
}

/// Byte size of the convert destination buffer for `params`.
#[inline]
pub fn dest_buffer_bytes(params: &ProcessingOptions) -> u32 {
    params
        .dest_format
        .frame_bytes(params.dest_w, params.dest_h)
}

/// Convert one destination pixel (storage-buffer semantics).
#[inline(always)]
pub fn convert_dest_pixel(src: &[u8], dst: &mut [u8], params: &ProcessingOptions, dx: u32, dy: u32) {
    let (sx, sy, valid) = map_dst_to_src(params, dx as f32, dy as f32);
    let (r, g, b) = if valid {
        sample_src(src, params, sx, sy)
    } else {
        (0.0, 0.0, 0.0)
    };
    write_dest_pixel(dst, params, dx, dy, r, g, b);
}

/// Full-frame convert into `dst` (must be at least [`dest_buffer_bytes`] long).
#[cfg(not(target_arch = "spirv"))]
pub fn convert_storage(src: &[u8], dst: &mut [u8], params: &ProcessingOptions) {
    for dy in 0..params.dest_h {
        for dx in 0..params.dest_w {
            convert_dest_pixel(src, dst, params, dx, dy);
        }
    }
}
