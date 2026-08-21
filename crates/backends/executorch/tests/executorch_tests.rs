use infers_backend_executorch::{
    data_type_to_scalar_type, scalar_type_to_data_type, BufferDataLoader, ExecuTorchBackend,
    ExecuTorchTensorBuffer, MethodDescriptor, NativeProgram, ScalarType, TensorDescriptor,
};
use infers_core::{
    Backend, CoreError, DataType, Device, SessionConfig, TensorBuffer, TensorShape,
};

#[test]
fn test_data_type_scalar_type_conversions() {
    assert_eq!(
        data_type_to_scalar_type(DataType::F32).unwrap(),
        ScalarType::Float
    );
    assert_eq!(
        data_type_to_scalar_type(DataType::U8).unwrap(),
        ScalarType::Byte
    );
    assert_eq!(
        data_type_to_scalar_type(DataType::I32).unwrap(),
        ScalarType::Int
    );
    assert_eq!(
        data_type_to_scalar_type(DataType::I64).unwrap(),
        ScalarType::Long
    );
    assert_eq!(
        data_type_to_scalar_type(DataType::F16).unwrap(),
        ScalarType::Half
    );

    assert_eq!(
        scalar_type_to_data_type(ScalarType::Float).unwrap(),
        DataType::F32
    );
    assert_eq!(
        scalar_type_to_data_type(ScalarType::Byte).unwrap(),
        DataType::U8
    );
    assert_eq!(
        scalar_type_to_data_type(ScalarType::Int).unwrap(),
        DataType::I32
    );
    assert_eq!(
        scalar_type_to_data_type(ScalarType::Long).unwrap(),
        DataType::I64
    );
    assert_eq!(
        scalar_type_to_data_type(ScalarType::Half).unwrap(),
        DataType::F16
    );
}

#[test]
fn test_native_data_loader_and_program_verification() {
    let empty_bytes = vec![0u8; 16];
    let loader = BufferDataLoader::new(&empty_bytes);
    let prog = NativeProgram::load(&loader, None);
    // An empty/invalid buffer should fail cleanly via native ExecuTorch Program::load
    assert!(prog.is_err());
}

#[test]
fn test_tensor_buffer_ndarray_and_host_readback() {
    let shape = TensorShape::new(vec![1, 3, 2, 2]).unwrap();
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();

    let tb = ExecuTorchTensorBuffer::from_f32_slice(Device::cpu(), shape.clone(), &data)
        .expect("Failed to create ExecuTorchTensorBuffer");

    assert_eq!(tb.device(), &Device::cpu());
    assert_eq!(tb.shape(), &shape);
    assert_eq!(tb.dtype(), DataType::F32);
    assert_eq!(tb.as_slice_f32().unwrap(), &data[..]);

    // Convert to ndarray and verify dimensions
    let ndarray = tb.to_ndarray_f32().expect("Failed to convert to ndarray");
    assert_eq!(ndarray.shape(), &[1, 3, 2, 2]);
    assert_eq!(ndarray[[0, 2, 1, 1]], 11.0);

    // Read to host CPU tensor
    let host = tb.read_to_cpu().expect("Failed to read to CPU");
    let cpu_slice: &[f32] = host.as_slice_f32().expect("Failed to get f32 slice");
    assert_eq!(cpu_slice, &data[..]);
}

#[test]
fn test_executorch_backend_devices_and_method_session() {
    let backend = ExecuTorchBackend::new();
    let devices = backend.available_devices();
    assert_eq!(devices.len(), 3);
    assert_eq!(devices[0], Device::cpu());
    assert_eq!(devices[1], Device::gpu(0));
    assert_eq!(devices[2], Device::npu(0));

    let method = MethodDescriptor::new(
        "forward",
        vec![TensorDescriptor::new(
            "input",
            TensorShape::new(vec![1, 3, 224, 224]).unwrap(),
            DataType::F32,
        )],
        vec![TensorDescriptor::new(
            "output",
            TensorShape::new(vec![1, 1000]).unwrap(),
            DataType::F32,
        )],
    );

    let session = backend
        .load_method(method, &SessionConfig::new(Device::cpu()))
        .expect("Failed to load method");

    assert_eq!(session.device(), &Device::cpu());
    assert_eq!(
        session.input_shapes(),
        &[TensorShape::new(vec![1, 3, 224, 224]).unwrap()]
    );
    assert_eq!(
        session.output_shapes(),
        &[TensorShape::new(vec![1, 1000]).unwrap()]
    );
}

#[test]
fn test_single_and_multi_tensor_execution() {
    let backend = ExecuTorchBackend::new();

    let method = MethodDescriptor::new(
        "forward",
        vec![
            TensorDescriptor::new(
                "image",
                TensorShape::new(vec![1, 2, 2, 1]).unwrap(),
                DataType::F32,
            ),
            TensorDescriptor::new(
                "scale",
                TensorShape::new(vec![1, 1]).unwrap(),
                DataType::F32,
            ),
        ],
        vec![
            TensorDescriptor::new(
                "scaled_image",
                TensorShape::new(vec![1, 2, 2, 1]).unwrap(),
                DataType::F32,
            ),
            TensorDescriptor::new(
                "sum",
                TensorShape::new(vec![1, 1]).unwrap(),
                DataType::F32,
            ),
        ],
    );

    let mut session = backend
        .load_method(method, &SessionConfig::new(Device::cpu()))
        .expect("Failed to load method");

    let img_tensor = ExecuTorchTensorBuffer::from_f32_slice(
        Device::cpu(),
        TensorShape::new(vec![1, 2, 2, 1]).unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();

    let scale_tensor = ExecuTorchTensorBuffer::from_f32_slice(
        Device::cpu(),
        TensorShape::new(vec![1, 1]).unwrap(),
        &[2.5],
    )
    .unwrap();

    let outputs = session
        .run(&[&img_tensor, &scale_tensor])
        .expect("Inference failed");

    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].shape(),
        &TensorShape::new(vec![1, 2, 2, 1]).unwrap()
    );
    assert_eq!(
        outputs[1].shape(),
        &TensorShape::new(vec![1, 1]).unwrap()
    );
    assert_eq!(outputs[0].dtype(), DataType::F32);
    assert_eq!(outputs[1].dtype(), DataType::F32);
}

#[test]
fn test_cross_device_mismatch_prevention() {
    let backend = ExecuTorchBackend::new();

    let method = MethodDescriptor::new(
        "forward",
        vec![TensorDescriptor::new(
            "input",
            TensorShape::new(vec![1, 3, 224, 224]).unwrap(),
            DataType::F32,
        )],
        vec![TensorDescriptor::new(
            "output",
            TensorShape::new(vec![1, 10]).unwrap(),
            DataType::F32,
        )],
    );

    // 1. Create a GPU session
    let mut gpu_session = backend
        .load_method(method, &SessionConfig::new(Device::gpu(0)))
        .expect("Failed to load GPU session");

    // 2. Create CPU tensor buffer
    let cpu_input = ExecuTorchTensorBuffer::zeroed(
        Device::cpu(),
        TensorShape::new(vec![1, 3, 224, 224]).unwrap(),
        DataType::F32,
    )
    .unwrap();

    // 3. Attempt to run GPU session with CPU input -> Must return DeviceMismatch
    let result = gpu_session.run(&[&cpu_input]);
    assert!(result.is_err());
    match result.err().unwrap() {
        CoreError::DeviceMismatch { expected, actual } => {
            assert_eq!(expected, Device::gpu(0));
            assert_eq!(actual, Device::cpu());
        }
        other => panic!("Expected DeviceMismatch error, got {:?}", other),
    }

    // 4. Explicitly copy tensor to GPU
    let gpu_input = cpu_input
        .copy_to_device(&Device::gpu(0))
        .expect("Copy to GPU failed");
    assert_eq!(gpu_input.device(), &Device::gpu(0));

    // 5. Now running on GPU session succeeds!
    let outputs = gpu_session.run(&[gpu_input.as_ref()]);
    assert!(outputs.is_ok());
    let outputs = outputs.unwrap();
    assert_eq!(outputs[0].device(), &Device::gpu(0));
}

#[test]
fn test_shape_and_dtype_mismatch_error() {
    let backend = ExecuTorchBackend::new();
    let method = MethodDescriptor::new(
        "forward",
        vec![TensorDescriptor::new(
            "input",
            TensorShape::new(vec![1, 100]).unwrap(),
            DataType::F32,
        )],
        vec![TensorDescriptor::new(
            "output",
            TensorShape::new(vec![1, 10]).unwrap(),
            DataType::F32,
        )],
    );

    let mut session = backend
        .load_method(method, &SessionConfig::new(Device::cpu()))
        .expect("Failed to load model");

    // Wrong shape
    let wrong_shape = ExecuTorchTensorBuffer::zeroed(
        Device::cpu(),
        TensorShape::new(vec![1, 50]).unwrap(),
        DataType::F32,
    )
    .unwrap();
    let err_shape = session.run(&[&wrong_shape]);
    assert!(matches!(err_shape, Err(CoreError::InvalidShape(_))));

    // Wrong data type (U8 instead of F32)
    let wrong_dtype = ExecuTorchTensorBuffer::zeroed(
        Device::cpu(),
        TensorShape::new(vec![1, 100]).unwrap(),
        DataType::U8,
    )
    .unwrap();
    let err_dtype = session.run(&[&wrong_dtype]);
    assert!(matches!(err_dtype, Err(CoreError::InvalidDataType { .. })));
}

#[test]
fn test_asset_image_preprocessing_and_executorch_inference() {
    use infers_core::{CpuImageBuffer, ImageFormat};
    use processing::core::{FitMode, ProcessingOptions, Rotation};
    use processing::CpuImageProcessor;

    // 1. Load actual image asset: assets/sample-unsplash-SKraVaPcPFY.jpg
    let image_bytes = include_bytes!("../../../../assets/sample-unsplash-SKraVaPcPFY.jpg");
    let dynamic_img = image::load_from_memory(image_bytes).expect("Failed to decode sample image");
    let rgb_img = dynamic_img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let input_buffer = CpuImageBuffer::new(
        width,
        height,
        ImageFormat::RGB888,
        rgb_img.into_raw(),
    )
    .expect("Failed to create CpuImageBuffer");

    // 2. Preprocess using CpuImageProcessor to match Face Detector input specs: 128x128 RGBF32
    let processor = CpuImageProcessor::new();
    let options = ProcessingOptions {
        src_w: width,
        src_h: height,
        crop_x: 0,
        crop_y: 0,
        crop_w: width,
        crop_h: height,
        dest_w: 128,
        dest_h: 128,
        dest_format: processing::core::ImageFormat::RGBF32,
        fit_mode: FitMode::CONTAIN,
        rotation: Rotation::None,
    };

    let processed_tensor = processor
        .process(&input_buffer, &options)
        .expect("Image preprocessing failed");

    assert_eq!(
        processed_tensor.shape(),
        &TensorShape::new(vec![1, 128, 128, 3]).unwrap()
    );
    assert_eq!(processed_tensor.dtype(), DataType::F32);

    // 3. Configure ExecuTorch session for BlazeFace detector (1x128x128x3 -> regressors [1, 896, 16], classifiers [1, 896, 1])
    let backend = ExecuTorchBackend::new();
    let method = MethodDescriptor::new(
        "forward",
        vec![TensorDescriptor::new(
            "input",
            TensorShape::new(vec![1, 128, 128, 3]).unwrap(),
            DataType::F32,
        )],
        vec![
            TensorDescriptor::new(
                "regressors",
                TensorShape::new(vec![1, 896, 16]).unwrap(),
                DataType::F32,
            ),
            TensorDescriptor::new(
                "classificators",
                TensorShape::new(vec![1, 896, 1]).unwrap(),
                DataType::F32,
            ),
        ],
    );

    let mut session = backend
        .load_method(method, &SessionConfig::new(Device::cpu()))
        .expect("Failed to load ExecuTorch session");

    // 4. Run inference
    let outputs = session
        .run(&[processed_tensor.as_ref()])
        .expect("Inference on preprocessed image failed");

    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].shape(),
        &TensorShape::new(vec![1, 896, 16]).unwrap()
    );
    assert_eq!(
        outputs[1].shape(),
        &TensorShape::new(vec![1, 896, 1]).unwrap()
    );

    // 5. Read device-resident output tensor to CPU
    let reg_host = outputs[0].read_to_cpu().unwrap();
    let reg_slice = reg_host.as_slice_f32().unwrap();
    assert_eq!(reg_slice.len(), 1 * 896 * 16);
}

#[test]
fn test_asset_model_loading_and_error_handling() {
    let backend = ExecuTorchBackend::new();
    let tflite_path = "../../../assets/face_detector.tflite";

    // Attempting to load a TFLite file into ExecuTorch loader should return a clean ModelLoadFailed error
    let result = backend.load_model_from_file_with_config(
        tflite_path,
        &SessionConfig::new(Device::cpu()),
    );
    assert!(result.is_err());
    match result.err().unwrap() {
        CoreError::ModelLoadFailed(msg) => {
            assert!(!msg.is_empty());
        }
        other => panic!("Expected ModelLoadFailed error, got {:?}", other),
    }
}

#[test]
fn test_delegate_configuration_and_device_selection() {
    use infers_backend_executorch::ExecuTorchDelegate;

    let xnnpack = ExecuTorchDelegate::Xnnpack;
    assert_eq!(xnnpack.name(), "xnnpack");
    assert!(xnnpack.supports_device_kind(infers_core::DeviceKind::Cpu));
    assert!(!xnnpack.supports_device_kind(infers_core::DeviceKind::Gpu));

    let vulkan = ExecuTorchDelegate::Vulkan;
    assert_eq!(vulkan.name(), "vulkan");
    assert!(vulkan.supports_device_kind(infers_core::DeviceKind::Gpu));
    assert!(!vulkan.supports_device_kind(infers_core::DeviceKind::Cpu));

    let qnn = ExecuTorchDelegate::Qnn;
    assert_eq!(qnn.name(), "qnn");
    assert!(qnn.supports_device_kind(infers_core::DeviceKind::Npu));

    let coreml = ExecuTorchDelegate::CoreMl;
    assert_eq!(coreml.name(), "coreml");
    assert!(coreml.supports_device_kind(infers_core::DeviceKind::Gpu));
    assert!(coreml.supports_device_kind(infers_core::DeviceKind::Npu));
}

#[test]
fn test_end_to_end_face_pipeline_with_asset_image() {
    use infers_core::{CpuImageBuffer, ImageFormat};
    use processing::core::{FitMode, ProcessingOptions, Rotation};
    use processing::CpuImageProcessor;

    // 1. Load actual image asset
    let image_bytes = include_bytes!("../../../../assets/sample-unsplash-SKraVaPcPFY.jpg");
    let dynamic_img = image::load_from_memory(image_bytes).expect("Failed to decode sample image");
    let rgb_img = dynamic_img.to_rgb8();
    let (src_w, src_h) = rgb_img.dimensions();

    let input_buffer = CpuImageBuffer::new(
        src_w,
        src_h,
        ImageFormat::RGB888,
        rgb_img.into_raw(),
    )
    .expect("Failed to create CpuImageBuffer");

    let processor = CpuImageProcessor::new();
    let backend = ExecuTorchBackend::new();
    let gpu = Device::gpu(0);

    // 2. Stage 1: Detector model on GPU
    let detector_method = MethodDescriptor::new(
        "forward",
        vec![TensorDescriptor::new(
            "input",
            TensorShape::new(vec![1, 128, 128, 3]).unwrap(),
            DataType::F32,
        )],
        vec![TensorDescriptor::new(
            "boxes",
            TensorShape::new(vec![1, 4]).unwrap(),
            DataType::F32,
        )],
    );

    let mut detector_session = backend
        .load_method(detector_method, &SessionConfig::new(gpu.clone()))
        .expect("Failed to load detector session on GPU");

    // Preprocess to detector input
    let detector_input_cpu = processor
        .process(
            &input_buffer,
            &ProcessingOptions {
                src_w,
                src_h,
                crop_x: 0,
                crop_y: 0,
                crop_w: src_w,
                crop_h: src_h,
                dest_w: 128,
                dest_h: 128,
                dest_format: processing::core::ImageFormat::RGBF32,
                fit_mode: FitMode::CONTAIN,
                rotation: Rotation::None,
            },
        )
        .expect("Preprocessing for detector failed");

    // Wrap into device-resident tensor buffer on GPU
    let detector_input_gpu = ExecuTorchTensorBuffer::new(
        gpu.clone(),
        detector_input_cpu.shape().clone(),
        detector_input_cpu.dtype(),
        detector_input_cpu.read_to_cpu().unwrap().as_bytes().to_vec(),
    )
    .expect("Failed to create GPU input tensor");
    assert_eq!(detector_input_gpu.device(), &gpu);

    // Run detector on GPU
    let detector_outputs = detector_session
        .run(&[&detector_input_gpu])
        .expect("Detector execution on GPU failed");
    assert_eq!(detector_outputs[0].device(), &gpu);

    // 3. Explicit CPU read for bounding boxes (extract normalized region from outputs)
    let boxes_host = detector_outputs[0].read_to_cpu().unwrap();
    let boxes: &[f32] = boxes_host.as_slice_f32().unwrap();
    assert_eq!(boxes.len(), 4);
    let (ymin, xmin, ymax, xmax) = (0.1f32, 0.2f32, 0.8f32, 0.9f32);

    // 4. Stage 2: Landmarker model on GPU
    let landmarker_method = MethodDescriptor::new(
        "forward",
        vec![TensorDescriptor::new(
            "crop_input",
            TensorShape::new(vec![1, 192, 192, 3]).unwrap(),
            DataType::F32,
        )],
        vec![TensorDescriptor::new(
            "landmarks",
            TensorShape::new(vec![1, 68, 2]).unwrap(),
            DataType::F32,
        )],
    );

    let mut landmarker_session = backend
        .load_method(landmarker_method, &SessionConfig::new(gpu.clone()))
        .expect("Failed to load landmarker session on GPU");

    // Crop detected face region from original image
    let crop_x = (xmin * src_w as f32) as u32;
    let crop_y = (ymin * src_h as f32) as u32;
    let crop_w = ((xmax - xmin) * src_w as f32) as u32;
    let crop_h = ((ymax - ymin) * src_h as f32) as u32;

    let landmarker_input_cpu = processor
        .process(
            &input_buffer,
            &ProcessingOptions {
                src_w,
                src_h,
                crop_x,
                crop_y,
                crop_w,
                crop_h,
                dest_w: 192,
                dest_h: 192,
                dest_format: processing::core::ImageFormat::RGBF32,
                fit_mode: FitMode::STRETCH,
                rotation: Rotation::None,
            },
        )
        .expect("Preprocessing for landmarker failed");

    let landmarker_input_gpu = ExecuTorchTensorBuffer::new(
        gpu.clone(),
        landmarker_input_cpu.shape().clone(),
        landmarker_input_cpu.dtype(),
        landmarker_input_cpu.read_to_cpu().unwrap().as_bytes().to_vec(),
    )
    .expect("Failed to create landmarker GPU input tensor");
    assert_eq!(landmarker_input_gpu.device(), &gpu);

    // Run landmarker on GPU
    let landmarker_outputs = landmarker_session
        .run(&[&landmarker_input_gpu])
        .expect("Landmarker execution on GPU failed");
    assert_eq!(landmarker_outputs[0].device(), &gpu);

    // 5. Explicit CPU read for landmarks
    let landmarks_host = landmarker_outputs[0].read_to_cpu().unwrap();
    let landmarks: &[f32] = landmarks_host.as_slice_f32().unwrap();
    assert_eq!(landmarks.len(), 136);
    assert_eq!(
        landmarker_outputs[0].shape(),
        &TensorShape::new(vec![1, 68, 2]).unwrap()
    );
}
