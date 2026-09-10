//! GPU-direct ExecuTorch Vulkan input path (Phase 1).

use crate::error::ExecuTorchError;
pub use crate::tensor_ptr::{evalue_to_cpu_tensor, OwnedTensorPtr};
use executorch::evalue::EValue;
use executorch::tensor::{TensorPtrBuilder, View};
use infers_core::{CoreError, DataType, Tensor, TensorShape};
use infers_gpu::{VulkanBufferHandle, VulkanContext};
use infers_gpu::Vulkan;
use std::sync::Arc;

#[link(name = "infers_et_vulkan_ffi")]
unsafe extern "C" {
    fn infers_et_take_registered_vulkan_graph() -> *mut std::ffi::c_void;
    fn infers_et_vulkan_set_skip_staging_copy_mask(mask: u64);
    fn infers_et_vulkan_input_staging_buffer(
        graph_ptr: *mut std::ffi::c_void,
        input_index: u32,
        out_buffer: *mut infers_gpu::ash::vk::Buffer,
        out_offset: *mut u64,
        out_size: *mut u64,
    ) -> i32;
}

/// Handle to a Vulkan-delegated model's `ComputeGraph`, captured after `load_method`.
///
/// Borrowed, not owned: the graph belongs to the ExecuTorch `Module` and this
/// handle is only valid while that module is loaded.
#[derive(Clone, Copy)]
pub struct VulkanComputeGraph {
    ptr: *mut std::ffi::c_void,
}

// SAFETY: the pointer refers to a `ComputeGraph` owned by the ExecuTorch module.
// Every use goes through `staging_target`, which only reads buffer metadata, and
// the session serialises calls behind the mutex guarding its module, so the
// graph is never touched concurrently through this handle.
unsafe impl Send for VulkanComputeGraph {}
unsafe impl Sync for VulkanComputeGraph {}

impl VulkanComputeGraph {
    /// Claim the graph registered by the most recent Vulkan `load_method`, if any.
    pub fn take_registered() -> Option<Self> {
        // SAFETY: returns either null or a pointer to the graph owned by the
        // module just loaded; the callee only reads a global set by the delegate.
        let ptr = unsafe { infers_et_take_registered_vulkan_graph() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn staging_target(
        &self,
        input_index: usize,
    ) -> Result<(VulkanBufferHandle, u64), ExecuTorchError> {
        let mut buffer = infers_gpu::ash::vk::Buffer::null();
        let mut offset = 0u64;
        let mut size = 0u64;
        // SAFETY: `self.ptr` is non-null and refers to a graph that is still
        // loaded (see the type's docs); the three out-parameters are valid,
        // uniquely borrowed locals. The callee reports failure via its return
        // code rather than unwinding.
        let rc = unsafe {
            infers_et_vulkan_input_staging_buffer(
                self.ptr,
                input_index as u32,
                &mut buffer,
                &mut offset,
                &mut size,
            )
        };
        if rc != 0 {
            return Err(ExecuTorchError::Execution(format!(
                "Failed to query ET-VK staging buffer for input {input_index}: code {rc}"
            )));
        }
        Ok((
            VulkanBufferHandle {
                buffer,
                memory: infers_gpu::ash::vk::DeviceMemory::null(),
                offset,
                size,
            },
            size,
        ))
    }
}

struct HostVisibleInput {
    buffer: infers_gpu::AllocatedBuffer,
    shape: TensorShape,
    dtype: DataType,
}

impl HostVisibleInput {
    fn new(context: &VulkanContext, shape: TensorShape, dtype: DataType) -> Result<Self, ExecuTorchError> {
        use infers_gpu::ash::vk;
        use infers_gpu::gpu_allocator::MemoryLocation;
        let bytes = shape.byte_size(dtype) as u64;
        let buffer = context
            .create_buffer(
                bytes,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                "et-host-visible-input",
            )
            .map_err(|err| ExecuTorchError::Execution(err.to_string()))?;
        Ok(Self {
            buffer,
            shape,
            dtype,
        })
    }

    fn copy_from_gpu(
        &self,
        context: &VulkanContext,
        src: VulkanBufferHandle,
    ) -> Result<(), ExecuTorchError> {
        // This buffer was sized to hold the whole tensor, so a shorter source
        // would leave part of the input undefined. Truncating the copy silently
        // would feed the model partial data, so reject it instead.
        require_transfer_bytes(self.buffer.size, src.size, "GPU input buffer")
            .map_err(|err| ExecuTorchError::Execution(err.to_string()))?;
        context
            .copy_buffer(src.buffer, src.offset, self.buffer.buffer, 0, self.buffer.size)
            .map_err(|err| ExecuTorchError::Execution(err.to_string()))
    }

    /// Build an ExecuTorch tensor that points directly at this buffer's mapping.
    ///
    /// The returned [`OwnedTensorPtr`] is typed `'static` because ExecuTorch has
    /// no lifetime parameter to thread the borrow through, so the pointer's
    /// validity is this module's responsibility: the returned value must not
    /// outlive `self`. [`GpuInputPlan`] upholds that by owning both and dropping
    /// the tensor pointers first.
    fn tensor_ptr(&self) -> Result<OwnedTensorPtr, CoreError> {
        let mapped = self.buffer.allocation.mapped_slice().ok_or_else(|| {
            CoreError::BufferTransferFailed("ET host-visible input is not mapped".into())
        })?;

        let needed = self.shape.byte_size(self.dtype);
        if mapped.len() < needed {
            return Err(CoreError::BufferTransferFailed(format!(
                "ET host-visible mapping is {} bytes, need {}",
                mapped.len(),
                needed
            )));
        }

        let dims: Vec<i32> = self.shape.dims().iter().map(|&d| d as i32).collect();
        match self.dtype {
            DataType::F32 => {
                let ptr = mapped.as_ptr();
                // Reading `f32` through a misaligned pointer is undefined
                // behaviour. Vulkan guarantees generous alignment for mapped
                // memory, but verify rather than assume.
                if !ptr.cast::<f32>().is_aligned() {
                    return Err(CoreError::BufferTransferFailed(
                        "ET host-visible mapping is not f32-aligned".into(),
                    ));
                }
                // SAFETY: `ptr` is the start of a mapping of at least `needed`
                // bytes (checked above), correctly aligned for `f32`, and stays
                // valid and unaliased while `self` lives. `dims` describes
                // exactly `needed` bytes, so ExecuTorch reads within the
                // mapping. See this method's doc comment for the lifetime
                // obligation the `'static` type erases.
                let ptr = unsafe {
                    TensorPtrBuilder::<View<f32>>::from_ptr(ptr.cast::<f32>(), dims)
                        .build()
                        .map_err(|e| {
                            CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}"))
                        })?
                };
                Ok(OwnedTensorPtr::F32(ptr))
            }
            DataType::U8 => {
                // SAFETY: as above; `u8` has no alignment requirement.
                let ptr = unsafe {
                    TensorPtrBuilder::<View<u8>>::from_ptr(mapped.as_ptr(), dims)
                        .build()
                        .map_err(|e| {
                            CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}"))
                        })?
                };
                Ok(OwnedTensorPtr::U8(ptr))
            }
            other => Err(CoreError::InferenceFailed(format!(
                "Unsupported GPU input dtype for ExecuTorch: {other:?}"
            ))),
        }
    }
}

/// Per-input slot for [`GpuInputPlan::evalues`].
enum PreparedInput {
    /// Staging already filled; patched `Method::set_input` ignores this evalue.
    SkipStaging,
    /// Host-visible fallback tensor (index into `GpuInputPlan::owned_ptrs`).
    Owned { owned_index: usize },
}

/// Prepared GPU inputs for one `Module::execute`.
///
/// Skip-staging slots use [`EValue::none`] — the patched runtime marks the
/// input set without copying, and the Vulkan delegate reads GPU staging.
/// Owned host-visible pointers may alias `host_fallback`'s mapped memory; the
/// [`Drop`] impl releases those pointers before the buffers they point into.
pub struct GpuInputPlan {
    prepared: Vec<PreparedInput>,
    owned_ptrs: Vec<OwnedTensorPtr>,
    pub skip_staging_mask: u64,
    host_fallback: Vec<HostVisibleInput>,
    _gpu_pins: Vec<Tensor<Vulkan>>,
}

impl Drop for GpuInputPlan {
    fn drop(&mut self) {
        // Drop the tensor pointers first: some alias `host_fallback`'s mappings,
        // which are unmapped when those buffers are freed.
        self.owned_ptrs.clear();
        self.host_fallback.clear();
    }
}

impl GpuInputPlan {
    /// Build ExecuTorch evalues for `execute`.
    ///
    /// Skip-staging inputs become [`EValue::none`]; the skip mask must already
    /// be installed so `Method::set_input` does not read them.
    pub fn evalues(&self) -> Vec<EValue<'_>> {
        self.prepared
            .iter()
            .map(|prep| match *prep {
                PreparedInput::SkipStaging => EValue::none(),
                PreparedInput::Owned { owned_index } => self.owned_ptrs[owned_index].as_evalue(),
            })
            .collect()
    }
}

pub fn prepare_inputs(
    inputs: &[&Tensor<Vulkan>],
    context: &Arc<VulkanContext>,
    vulkan_graph: Option<VulkanComputeGraph>,
) -> Result<GpuInputPlan, CoreError> {
    let mut prepared = Vec::with_capacity(inputs.len());
    let mut owned_ptrs = Vec::new();
    let mut skip_staging_mask = 0u64;
    let mut host_fallback = Vec::new();
    let mut gpu_pins = Vec::new();

    for (index, input) in inputs.iter().enumerate() {
        if !Arc::ptr_eq(context, input.device().context()) {
            return Err(CoreError::DeviceMismatch {
                expected: context.device_info().clone(),
                actual: input.device().info().clone(),
            });
        }

        let src = input.storage().vulkan_handle().ok_or_else(|| {
            CoreError::BufferTransferFailed("GPU tensor buffer already destroyed".into())
        })?;
        gpu_pins.push((*input).clone());

        let needed = input.shape().byte_size(input.dtype()) as u64;

        let staging = if index < u64::BITS as usize {
            vulkan_graph.and_then(|graph| graph.staging_target(index).ok())
        } else {
            None
        };

        if let Some((dst, staging_size)) = staging {
            require_transfer_bytes(needed, src.size, &format!("input {index} source"))?;
            require_transfer_bytes(
                needed,
                staging_size,
                &format!("input {index} ET-VK staging"),
            )?;
            // Preprocess with `MaterializeTarget::SessionInput` already wrote
            // into this staging buffer; a self-copy is redundant and overlapping
            // `vkCmdCopyBuffer` regions are undefined.
            let already_in_staging =
                src.buffer == dst.buffer && src.offset == dst.offset;
            if !already_in_staging {
                context
                    .copy_buffer(src.buffer, src.offset, dst.buffer, dst.offset, needed)
                    .map_err(|err| CoreError::BufferTransferFailed(err.to_string()))?;
            }
            skip_staging_mask |= 1u64 << index;
            prepared.push(PreparedInput::SkipStaging);
            continue;
        }

        let host = HostVisibleInput::new(context, input.shape().clone(), input.dtype())?;
        host.copy_from_gpu(context, src)
            .map_err(|err| CoreError::BufferTransferFailed(err.to_string()))?;
        let owned_index = owned_ptrs.len();
        owned_ptrs.push(host.tensor_ptr()?);
        host_fallback.push(host);
        prepared.push(PreparedInput::Owned { owned_index });
    }

    Ok(GpuInputPlan {
        prepared,
        owned_ptrs,
        skip_staging_mask,
        host_fallback,
        _gpu_pins: gpu_pins,
    })
}

/// Tell the patched Vulkan delegate / `Method::set_input` which inputs already
/// have their staging buffers filled, so both skip their host copies.
///
/// The mask is thread-local on the C++ side, so this must be called on the same
/// thread that goes on to run `execute`.
pub fn set_skip_staging_copy_mask(mask: u64) {
    // SAFETY: the callee only stores `mask` in a thread-local and cannot fail or
    // unwind.
    unsafe { infers_et_vulkan_set_skip_staging_copy_mask(mask) };
}

fn require_transfer_bytes(
    needed: u64,
    available: u64,
    label: &str,
) -> Result<(), CoreError> {
    if available < needed {
        Err(CoreError::BufferTransferFailed(format!(
            "{label} needs {needed} bytes but only {available} are available"
        )))
    } else {
        Ok(())
    }
}
