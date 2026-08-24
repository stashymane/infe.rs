use processing::{FitMode, ImageFormat, ProcessingOptions, Rotation, SHADERS};

#[test]
fn test_spirv_shader_binary_valid_and_entry_points() {
    assert!(!SHADERS.is_empty(), "Compiled SPIR-V should not be empty");
    assert!(
        SHADERS.len() >= 20,
        "SPIR-V header is at least 5 32-bit words (20 bytes)"
    );

    // SPIR-V magic number is 0x07230203 (little endian bytes: 0x03, 0x02, 0x23, 0x07)
    let magic = u32::from_le_bytes([SHADERS[0], SHADERS[1], SHADERS[2], SHADERS[3]]);
    assert_eq!(
        magic, 0x07230203,
        "SPIR-V binary magic number mismatch: expected 0x07230203, got 0x{:08x}",
        magic
    );

    // Parse SPIR-V words to verify entry points
    let words: Vec<u32> = SHADERS
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut entry_points = Vec::new();
    let mut i = 5; // Skip 5-word SPIR-V header
    while i < words.len() {
        let instr = words[i];
        let opcode = instr & 0xffff;
        let word_count = (instr >> 16) as usize;
        if word_count == 0 {
            break;
        }

        // OpEntryPoint = 15
        if opcode == 15 && word_count >= 4 {
            let name_words = &words[i + 3..i + word_count];
            let mut name_bytes = Vec::new();
            'outer: for &w in name_words {
                for b in w.to_le_bytes() {
                    if b == 0 {
                        break 'outer;
                    }
                    name_bytes.push(b);
                }
            }
            if let Ok(name) = String::from_utf8(name_bytes) {
                entry_points.push(name);
            }
        }

        i += word_count;
    }

    assert!(
        entry_points.contains(&"convert_main".to_string()),
        "SPIR-V module should have convert_main entry point, found: {:?}",
        entry_points
    );
    assert!(
        entry_points.contains(&"convert_image".to_string()),
        "SPIR-V module should have convert_image entry point, found: {:?}",
        entry_points
    );
}

#[test]
fn test_processing_options_enums() {
    assert_eq!(FitMode::STRETCH as u32, 0);
    assert_eq!(FitMode::CONTAIN as u32, 1);
    assert_eq!(FitMode::CROP as u32, 2);

    assert_eq!(ImageFormat::RGB888 as u32, 0);
    assert_eq!(ImageFormat::RGBF32 as u32, 1);
    assert_eq!(ImageFormat::NV12 as u32, 2);
    assert_eq!(ImageFormat::I420 as u32, 3);

    assert_eq!(Rotation::None as u32, 0);
    assert_eq!(Rotation::R90DEG as u32, 1);
    assert_eq!(Rotation::R180DEG as u32, 2);
    assert_eq!(Rotation::R270DEG as u32, 3);
}

#[test]
fn test_processing_options_crop() {
    let opts_no_crop = ProcessingOptions {
        src_w: 1920,
        src_h: 1080,
        crop_x: 0,
        crop_y: 0,
        crop_w: 0,
        crop_h: 0,
        dest_w: 224,
        dest_h: 224,
        dest_format: ImageFormat::RGBF32,
        fit_mode: FitMode::CONTAIN,
        rotation: Rotation::None,
        ..Default::default()
    };

    assert_eq!(opts_no_crop.effective_crop(), (0, 0, 1920, 1080));

    let opts_with_crop = ProcessingOptions {
        src_w: 1920,
        src_h: 1080,
        crop_x: 100,
        crop_y: 50,
        crop_w: 400,
        crop_h: 300,
        dest_w: 224,
        dest_h: 224,
        dest_format: ImageFormat::RGB888,
        fit_mode: FitMode::CROP,
        rotation: Rotation::R90DEG,
        ..Default::default()
    };

    assert_eq!(opts_with_crop.effective_crop(), (100, 50, 400, 300));
}
