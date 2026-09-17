use crate::interned_ir::InternedVal;
use pyo3::prelude::*;

trait GenericIndex {
    fn size(&self) -> Option<i64>;
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct ForwardSlice{
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
}


impl GenericIndex for ForwardSlice {
    fn size(&self) -> Option<i64> {
        match (self.start, self.stop, self.step) {
            // range(start, stop, step)
            (Some(start), Some(stop), Some(step)) => {
                if step == 0 {
                    None
                } else if step > 0 {
                    Some((stop - start + step - 1) / step)
                } else {
                    Some((start - stop - step - 1) / (-step))
                }
            },
            // range(start, stop)
            (Some(start), Some(stop), None) => {
                if stop >= start {
                    Some(stop - start)
                } else {
                    None
                }
            },
            // range(stop) 
            (None, Some(stop), None) => Some(stop),
            // Can't determine
            _ => None,
        }
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Scalar{
    value: i64,
    drops_dim: bool,
    position: Option<i64>
}


#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Slice {
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct GeneralSlice{
    value: Slice
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Positions{
    values: Vec<i64>
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Mask{
    values: Vec<bool>
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Label{
    value: InternedVal
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Advanced{
    dims: Vec<InternedVal>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indexer {
    Scalar(Scalar),
    ForwardSlice(ForwardSlice),
    GeneralSlice(GeneralSlice),
    Positions(Positions),
    Mask(Mask),
    Label(Label),
    Advanced(Advanced),
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
