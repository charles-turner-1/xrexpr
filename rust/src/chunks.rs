use crate::payload::PayloadRef;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// One dim's chunk spec: the Rust side of `xrexpr.chunks.ChunkSpec`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkSpec {
    SingleSize(i32),
    Auto,
    ByteSize(String),
    FullDim,
    NoChange,
    BlockSeq(Vec<i64>),
    /// A spec nothing models. Its value is a payload: Rust can carry it, not read it.
    OpaqueChunk(PayloadRef),
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
            "OpaqueChunk" => Ok(ChunkSpec::OpaqueChunk(obj.getattr("value")?.extract()?)),
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Cannot convert object of type {} to ChunkSpec",
                ty
            ))),
        }
    }
}

/// Converts to `(variant, fields)`, e.g. `("SingleSize", {"size": 10})` or
/// `("Auto", {})`, which `xrexpr.intern.crossing.from_rust` rebuilds into the Python spec.
/// The field names are the Python dataclass fields.
impl<'py> IntoPyObject<'py> for ChunkSpec {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let fields = PyDict::new(py);
        let variant = match self {
            ChunkSpec::SingleSize(size) => {
                fields.set_item("size", size)?;
                "SingleSize"
            }
            ChunkSpec::Auto => "Auto",
            ChunkSpec::ByteSize(value) => {
                fields.set_item("value", value)?;
                "ByteSize"
            }
            ChunkSpec::FullDim => "FullDim",
            ChunkSpec::NoChange => "NoChange",
            ChunkSpec::BlockSeq(sizes) => {
                fields.set_item("sizes", sizes)?;
                "BlockSeq"
            }
            ChunkSpec::OpaqueChunk(value) => {
                fields.set_item("value", value)?;
                "OpaqueChunk"
            }
        };
        (variant, fields).into_pyobject(py)
    }
}
