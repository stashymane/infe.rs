use crate::error::ExecuTorchError;
pub use executorch::ndarray;
use executorch::tensor::ScalarType;
use infers_core::DataType;

pub fn data_type_to_scalar_type(dt: DataType) -> Result<ScalarType, ExecuTorchError> {
    match dt {
        DataType::U8 => Ok(ScalarType::Byte),
        DataType::I8 => Ok(ScalarType::Char),
        DataType::I16 => Ok(ScalarType::Short),
        DataType::I32 => Ok(ScalarType::Int),
        DataType::I64 => Ok(ScalarType::Long),
        DataType::F16 => Ok(ScalarType::Half),
        DataType::F32 => Ok(ScalarType::Float),
        DataType::F64 => Ok(ScalarType::Double),
    }
}

pub fn scalar_type_to_data_type(st: ScalarType) -> Result<DataType, ExecuTorchError> {
    match st {
        ScalarType::Byte => Ok(DataType::U8),
        ScalarType::Char => Ok(DataType::I8),
        ScalarType::Short => Ok(DataType::I16),
        ScalarType::Int => Ok(DataType::I32),
        ScalarType::Long => Ok(DataType::I64),
        ScalarType::Half => Ok(DataType::F16),
        ScalarType::Float => Ok(DataType::F32),
        ScalarType::Double => Ok(DataType::F64),
        other => Err(ExecuTorchError::Execution(format!(
            "Unsupported ExecuTorch ScalarType: {:?}",
            other
        ))),
    }
}
