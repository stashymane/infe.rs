use crate::consumed::ConsumedFlag;
use crate::device::Device;
use crate::error::CoreError;
use crate::tensor::{DataType, Tensor, TensorShape};

/// Where a [`Pending`] commit writes its tensor output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaterializeTarget {
    #[default]
    Owned,
    SessionInput {
        slot: usize,
    },
}

type MaterializeFn<D> = Box<dyn FnOnce(MaterializeTarget) -> Result<Tensor<D>, CoreError> + Send>;

/// Unmaterialized preprocess/tensor work on device [D].
pub struct Pending<D: Device> {
    pub(crate) device: D,
    shape: TensorShape,
    dtype: DataType,
    materialize: MaterializeFn<D>,
    consumed: ConsumedFlag,
}

impl<D: Device> Pending<D> {
    pub fn schedule(
        device: D,
        shape: TensorShape,
        dtype: DataType,
        materialize: MaterializeFn<D>,
    ) -> Self {
        Self {
            device,
            shape,
            dtype,
            materialize,
            consumed: ConsumedFlag::default(),
        }
    }

    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    pub fn materialize(self) -> Result<Tensor<D>, CoreError> {
        self.materialize_with_target(MaterializeTarget::Owned)
    }

    pub fn materialize_with_target(self, target: MaterializeTarget) -> Result<Tensor<D>, CoreError> {
        self.consumed.take()?;
        (self.materialize)(target)
    }
}
