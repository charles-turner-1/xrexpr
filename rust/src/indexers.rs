use crate::interned_ir::InternedVal;
use std::num::{NonZeroU64, NonZeroI64};
use pyo3::prelude::*;

/// A trait for generic indexing operations.
/// `current` is the current size of the dimension being indexed. 
trait GenericIndex {
    fn size(&self, current: i64) -> Option<i64>;
}

/// Stolen from cpython/Objects/sliceobject.c#L256-296
fn adjust_indices(mut start: i64, mut stop: i64, step: NonZeroI64, length: i64) -> i64 {
    // No need to check step != 0, nor step >= -Py_SSIZE_T_MAX, since NonZeroI64 guarantees that step != 0 and step >= -i64::MAX.

    if start < 0 {
        start += length;
        if start < 0 {
            start = if step.get() < 0 { -1 } else {0};
        }
    } else if start >= length {
        start = if step.get() < 0 {length -1} else {length};
    }

    if stop < 0 {
        stop += length;
        if stop < 0 {
            stop = if step.get() < 0 { -1} else {0};
        }
    } else if stop >= length {
        stop = if step.get() < 0 {length -1} else {length};
    }

    if step.get() < 0 {
        if stop < start {
            return ((start - stop - 1) / -step.get()) + 1
        }
    } else {
        if start < stop {
            return ((stop - start - 1) / step.get()) + 1
        }
    }
    return 0

}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
struct Slice {
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<NonZeroI64>,
}


#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct ForwardSlice{
    start: Option<u64>,
    stop: Option<u64>,
    step: Option<NonZeroU64>, // Must be a positive integer
}

impl GenericIndex for ForwardSlice {
    fn size(&self, current: i64) -> Option<i64> {
        let step = self.step.map(|s| s.get() as i64).unwrap_or(1);
        let start = self.start.map(|s| s as i64).unwrap_or(0);
        let stop = self.stop.map(|s| s as i64).unwrap_or(current);

        Some(adjust_indices(start, stop, NonZeroI64::new(step).unwrap(), current))
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Scalar{
    value: i64,
    drops_dim: bool,
    position: Option<i64>
}

impl GenericIndex for Scalar {
    fn size(&self, _current: i64) -> Option<i64> {
        panic!("a scalar indexer drops its dim; its size is undefined");
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct GeneralSlice{
    value: Slice
}

impl GenericIndex for GeneralSlice {
    fn size(&self, current: i64) -> Option<i64> {
        let step = self.value.step.map(|s| s.get() as i64).unwrap_or(1);
        let start = self.value.start.map(|s| s as i64).unwrap_or(if step > 0 {0} else {current -1});
        let stop = self.value.stop.map(|s| s as i64).unwrap_or(if step > 0 {current} else {-1});

        Some(adjust_indices(start, stop, NonZeroI64::new(step).unwrap(), current))
    }
}


#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Positions{
    values: Vec<i64>
}

impl GenericIndex for Positions {
    fn size(&self, _current: i64) -> Option<i64> {
        Some(self.values.len() as i64)
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Mask{
    values: Vec<bool>
}

impl GenericIndex for Mask {
    fn size(&self, _current: i64) -> Option<i64> {
        Some(self.values.iter().filter(|&&v| v).count() as i64)
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Label{
    #[pyo3(attribute("known_size"))]
    size: Option<i64>,
}

impl GenericIndex for Label {
    fn size(&self, _current: i64) -> Option<i64> {
        self.size
    }
}

#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Advanced{
    dims: Vec<InternedVal>
}

impl GenericIndex for Advanced {
    fn size(&self, _current: i64) -> Option<i64> {
        panic!("an advanced indexer is never emitted; its select is Opaque");
    }
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

impl Indexer {
    pub fn size(&self, current: i64) -> Option<i64> {
        match self {
            Indexer::ForwardSlice(s) => s.size(current),
            Indexer::GeneralSlice(s) => s.size(current),
            Indexer::Positions(s) => s.size(current),
            Indexer::Mask(s) => s.size(current),
            Indexer::Label(s) => s.size(current),
            Indexer::Scalar(s) => s.size(current),
            Indexer::Advanced(s) => s.size(current),
        }
    }
}

impl FromPyObject<'_, '_> for Indexer {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> Result<Self, Self::Error> {
        let ty = obj.get_type().name()?;
        match ty.to_str()? {
            "Scalar" => Ok(Indexer::Scalar(obj.extract()?)),
            "ForwardSlice" => Ok(Indexer::ForwardSlice(obj.extract()?)),
            "GeneralSlice" => Ok(Indexer::GeneralSlice(obj.extract()?)),
            "Positions" => Ok(Indexer::Positions(obj.extract()?)),
            "Mask" => Ok(Indexer::Mask(obj.extract()?)),
            "Label" => Ok(Indexer::Label(obj.extract()?)),
            "Advanced" => Ok(Indexer::Advanced(obj.extract()?)),
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Cannot convert object of type {} to Indexer", ty
            ))),
        }
    }
}
