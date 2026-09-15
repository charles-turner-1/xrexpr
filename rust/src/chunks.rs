use pyo3::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkSpec {
    SingleSize(i32),
    Auto,
    ByteSize(String),
    FullDim,
    NoChange,
    BlockSeq(Vec<i64>),
    OpaqueChunk,
}

impl FromPyObject<'_, '_> for ChunkSpec {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> Result<Self, Self::Error> {
        let ty = obj.get_type().name()?;
        match ty.to_str()? {
            "SingleSize" => Ok(ChunkSpec::SingleSize(obj.getattr("size")?.extract()?)),
            "Auto" => Ok(ChunkSpec::Auto),
            "ByteSize" => Ok(ChunkSpec::ByteSize(obj.getattr("value")?.extract()?)),
            "FullDim" => Ok(ChunkSpec::FullDim),
            "NoChange" => Ok(ChunkSpec::NoChange),
            "BlockSeq" => Ok(ChunkSpec::BlockSeq(obj.getattr("sizes")?.extract()?)),
            "OpaqueChunk" => Ok(ChunkSpec::OpaqueChunk),
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Cannot convert object of type {} to ChunkSpec", ty
            ))),
        }
    }
}

