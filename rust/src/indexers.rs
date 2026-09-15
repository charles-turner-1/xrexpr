use crate::interned_ir::InternedVal;
use pyo3::prelude::*;

#[derive(FromPyObject)]
pub struct Scalar{
    value: i64,
    drops_dims: bool,
    position: Option<i64>
}

#[derive(FromPyObject)]
pub struct ForwardSlice{
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
}

#[derive(FromPyObject, Clone)]
pub struct Slice {
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
}

#[derive(FromPyObject)]
pub struct GeneralSlice{
    value: Slice
}

#[derive(FromPyObject)]
pub struct Positions{
    values: Vec<i64>
}

#[derive(FromPyObject)]
pub struct Mask{
    values: Vec<bool>
}

#[derive(FromPyObject)]
pub struct Label{
    value: InternedVal
}

#[derive(FromPyObject)]
pub struct Advanced{
    dims: Vec<InternedVal>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indexer {
    Scalar,
    ForwardSlice,
    GeneralSlice,
    Positions,
    Mask,
    Label,
    Advanced
}

impl FromPyObject<'_, '_> for Indexer {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> Result<Self, Self::Error> {
        let ty = obj.get_type().name()?;
        match ty.to_str()? {
            "Scalar" => Ok(Indexer::Scalar),
            "ForwardSlice" => Ok(Indexer::ForwardSlice),
            "GeneralSlice" => Ok(Indexer::GeneralSlice),
            "Positions" => Ok(Indexer::Positions),
            "Mask" => Ok(Indexer::Mask),
            "Label" => Ok(Indexer::Label),
            "Advanced" => Ok(Indexer::Advanced),
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Cannot convert object of type {} to Indexer", ty
            ))),
        }
    }
}
