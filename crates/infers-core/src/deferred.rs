use crate::consumed::ConsumedFlag;
use crate::device::Device;
use crate::error::CoreError;
use crate::image::HardwareImage;
use std::sync::Arc;

/// Recipe describing future device placement without committing work.
enum DeferredImageRecipe<D: Device> {
    FromHardware(Arc<HardwareImage>),
    Custom(Box<dyn FnOnce(&D) -> Result<D::Image, CoreError> + Send>),
}

/// Unmaterialized image on device [D]. No compute or transfer has run yet.
pub struct Deferred<D: Device> {
    device: D,
    recipe: DeferredImageRecipe<D>,
    consumed: ConsumedFlag,
}

impl<D: Device> Deferred<D> {
    pub fn from_hardware(device: D, hardware: Arc<HardwareImage>) -> Self {
        Self {
            device,
            recipe: DeferredImageRecipe::FromHardware(hardware),
            consumed: ConsumedFlag::default(),
        }
    }

    pub fn from_custom(
        device: D,
        materialize: impl FnOnce(&D) -> Result<D::Image, CoreError> + Send + 'static,
    ) -> Self {
        Self {
            device,
            recipe: DeferredImageRecipe::Custom(Box::new(materialize)),
            consumed: ConsumedFlag::default(),
        }
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    pub fn is_hardware(&self) -> bool {
        matches!(self.recipe, DeferredImageRecipe::FromHardware(_))
    }

    pub fn into_hardware(self) -> Result<(D, Arc<HardwareImage>), CoreError> {
        self.consumed.take()?;
        match self.recipe {
            DeferredImageRecipe::FromHardware(hardware) => Ok((self.device, hardware)),
            DeferredImageRecipe::Custom(_) => Err(CoreError::InvalidArgument(
                "deferred image is not sourced from hardware bytes".into(),
            )),
        }
    }

    pub fn materialize(self) -> Result<D::Image, CoreError> {
        self.consumed.take()?;
        match self.recipe {
            DeferredImageRecipe::FromHardware(hardware) => {
                D::materialize_deferred(&self.device, hardware)
            }
            DeferredImageRecipe::Custom(custom) => custom(&self.device),
        }
    }
}
