use pyo3::prelude::*;
use pyo3::types::PyTuple;

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

/// Rust's *view* of a chunk spec, as a `(variant, payload)` tuple: `("SingleSize", 10)`,
/// `("Auto", None)`. Not a `ChunkSpec` rebuilt -- `OpaqueChunk` keeps no payload here, so
/// there is nothing to rebuild it from. Used by the boundary test (`interned_ir::view_*`).
impl<'py> IntoPyObject<'py> for ChunkSpec {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        // Each arm's payload is a different Rust type, so each is converted to a Python
        // object (`Bound<PyAny>`) first, giving every arm the same tuple type.
        let (variant, payload) = match self {
            ChunkSpec::SingleSize(size) => ("SingleSize", size.into_pyobject(py)?.into_any()),
            ChunkSpec::Auto => ("Auto", py.None().into_bound(py)),
            ChunkSpec::ByteSize(value) => ("ByteSize", value.into_pyobject(py)?.into_any()),
            ChunkSpec::FullDim => ("FullDim", py.None().into_bound(py)),
            ChunkSpec::NoChange => ("NoChange", py.None().into_bound(py)),
            ChunkSpec::BlockSeq(sizes) => ("BlockSeq", sizes.into_pyobject(py)?.into_any()),
            ChunkSpec::OpaqueChunk => ("OpaqueChunk", py.None().into_bound(py)),
        };
        (variant, payload).into_pyobject(py)
    }
}
