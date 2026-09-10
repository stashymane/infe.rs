#![no_std]

use infers_processing_core::{
    ProcessingOptions, map_dst_to_src, sample_src, store_f32_le, write_dest_pixel,
};
use spirv_std::glam::{UVec3, Vec2};
use spirv_std::image::SampledImage;
use spirv_std::{Image, spirv};

/// Storage-buffer convert path. `params.src_format` selects RGB888, RGBF32, NV12, or I420.
#[spirv(compute(threads(16, 16)))]
pub fn convert_main(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] params: &ProcessingOptions,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] src: &[u8],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] dst: &mut [u8],
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
    #[spirv(push_constant)] params: &ProcessingOptions,
    #[spirv(descriptor_set = 0, binding = 0)] src: &SampledImage<Image!(2D, type = f32, sampled)>,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] dst: &mut [u8],
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

#[repr(C)]
pub struct LayoutDims {
    h: u32,
    w: u32,
}

fn load_f32_le(src: &[u8], offset: usize) -> f32 {
    if offset + 4 > src.len() {
        return 0.0;
    }
    let bits = (src[offset] as u32)
        | ((src[offset + 1] as u32) << 8)
        | ((src[offset + 2] as u32) << 16)
        | ((src[offset + 3] as u32) << 24);
    f32::from_bits(bits)
}

/// Transpose `[1, H, W, 3]` f32 NHWC into `[1, 3, H, W]` NCHW on GPU.
#[spirv(compute(threads(16, 16)))]
pub fn nhwc_to_nchw_main(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] dims: &LayoutDims,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] src: &[u8],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] dst: &mut [u8],
) {
    let dx = gid.x;
    let dy = gid.y;
    if dx >= dims.w || dy >= dims.h {
        return;
    }
    let hw = (dims.w * dims.h) as usize;
    let i = (dy * dims.w + dx) as usize;
    let base = i * 12;
    let r = load_f32_le(src, base);
    let g = load_f32_le(src, base + 4);
    let b = load_f32_le(src, base + 8);
    store_f32_le(dst, i * 4, r);
    store_f32_le(dst, (hw + i) * 4, g);
    store_f32_le(dst, (2 * hw + i) * 4, b);
}
