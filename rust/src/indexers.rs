use crate::interned_ir::InternedVal;
use crate::payload::PayloadRef;
use std::num::{NonZeroU64, NonZeroI64};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

/// A trait for generic indexing operations.
/// `current` is the current size of the dimension being indexed. 
trait GenericIndex {
    fn size(&self, current: i64) -> Option<i64>;
}

/// `len(range(length)[start:stop:step])`, as CPython computes it in `PySlice_Unpack` and
/// `PySlice_AdjustIndices` (Objects/sliceobject.c). CPython encodes a missing bound as a
/// sentinel and clamps it back into range; here `None` resolves straight to its end.
/// Equivalence is checked against CPython itself by the proptests below.
fn slice_len(start: Option<i64>, stop: Option<i64>, step: Option<NonZeroI64>, length: i64) -> i64 {
    let step = step.map_or(1, NonZeroI64::get);
    // The range a resolved index can take: `-1` is "before the first element" going backward.
    let (first, last) = if step < 0 { (-1, length - 1) } else { (0, length) };
    // Negative indices count from the end; `b + length` can't overflow as `b < 0 <= length`.
    let resolve = |b: i64| (if b < 0 { b + length } else { b }).clamp(first, last);

    let start = start.map_or(if step < 0 { last } else { first }, resolve);
    let stop = stop.map_or(if step < 0 { first } else { last }, resolve);
    let span = if step < 0 { start - stop } else { stop - start };
    // `unsigned_abs` takes i64::MIN to 2^63, where negating the step would overflow; the
    // quotient is at most `span`, so it fits back in an i64.
    if span <= 0 { 0 } else { (span as u64).div_ceil(step.unsigned_abs()) as i64 }
}

/// A Python `slice` with integer bounds, read from its `start`/`stop`/`step` attributes.
#[derive(FromPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
struct Slice {
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<NonZeroI64>,
}

/// Converts back to a real Python `slice`, which is what `GeneralSlice.value` holds.
impl<'py> IntoPyObject<'py> for Slice {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        py.get_type::<PySlice>()
            .call1((self.start, self.stop, self.step.map(NonZeroI64::get)))
    }
}


#[derive(FromPyObject, IntoPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct ForwardSlice{
    start: Option<u64>,
    stop: Option<u64>,
    step: Option<NonZeroU64>, // Must be a positive integer
}

impl GenericIndex for ForwardSlice {
    fn size(&self, current: i64) -> Option<i64> {
        // A u64 above i64::MAX saturates, as CPython clamps an oversized slice index to
        // PY_SSIZE_T_MAX; an `as` cast would wrap it negative and count from the end.
        let clamp = |b: u64| i64::try_from(b).unwrap_or(i64::MAX);
        let step = self.step.map(|s| NonZeroI64::new(clamp(s.get())).unwrap());
        Some(slice_len(self.start.map(clamp), self.stop.map(clamp), step, current))
    }
}

/// A single position or label, which drops its dim.
///
/// Half semantic, half payload. An integer position (`isel(time=3)`) is data a rule can
/// compose. A coordinate label (`sel(time="2020")`) could be any Python object, so Python
/// swaps it for a [`PayloadRef`] before crossing, and Rust can only carry it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scalar {
    Position(i64),
    Label(PayloadRef),
}

impl Scalar {
    /// The integer position, or `None` for a label.
    pub fn position(&self) -> Option<i64> {
        match self {
            Scalar::Position(p) => Some(*p),
            Scalar::Label(_) => None,
        }
    }
}

/// Reads Python's `Scalar.value`: a `PayloadRef` is a label, an `int` is a position.
/// Anything else is a label Python forgot to swap out, and is rejected.
impl FromPyObject<'_, '_> for Scalar {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let value = obj.getattr("value")?;
        if let Ok(label) = value.extract::<PayloadRef>() {
            return Ok(Scalar::Label(label));
        }
        value.extract::<i64>().map(Scalar::Position).map_err(|_| {
            PyTypeError::new_err("a Scalar's value must be an int position or a PayloadRef")
        })
    }
}

impl GenericIndex for Scalar {
    fn size(&self, _current: i64) -> Option<i64> {
        panic!("a scalar indexer drops its dim; its size is undefined");
    }
}

#[derive(FromPyObject, IntoPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct GeneralSlice{
    value: Slice
}

impl GenericIndex for GeneralSlice {
    fn size(&self, current: i64) -> Option<i64> {
        Some(slice_len(self.value.start, self.value.stop, self.value.step, current))
    }
}


#[derive(FromPyObject, IntoPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Positions{
    values: Vec<i64>
}

impl GenericIndex for Positions {
    fn size(&self, _current: i64) -> Option<i64> {
        Some(self.values.len() as i64)
    }
}

#[derive(FromPyObject, IntoPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct Mask{
    values: Vec<bool>
}

impl GenericIndex for Mask {
    fn size(&self, _current: i64) -> Option<i64> {
        Some(self.values.iter().filter(|&&v| v).count() as i64)
    }
}

/// A coordinate label, label slice or label sequence. The labels themselves are a payload;
/// Rust only sees how many there are, when that's known.
#[derive(FromPyObject, IntoPyObject, Debug, Clone, PartialEq, Eq)]
pub struct Label{
    value: PayloadRef,
    known_size: Option<i64>,
}

impl GenericIndex for Label {
    fn size(&self, _current: i64) -> Option<i64> {
        self.known_size
    }
}

#[derive(FromPyObject, IntoPyObject, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
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

/// Converts to `(variant, fields)`, e.g. `("Positions", {"values": [1, 3]})`, which
/// `xrexpr.intern.crossing.from_rust` rebuilds into the Python indexer.
impl<'py> IntoPyObject<'py> for Indexer {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        // Each inner struct derives `IntoPyObject`, which turns it into a dict of its fields.
        match self {
            Indexer::Scalar(s) => {
                let fields = PyDict::new(py);
                match s {
                    Scalar::Position(p) => fields.set_item("value", p)?,
                    Scalar::Label(label) => fields.set_item("value", label)?,
                }
                ("Scalar", fields).into_pyobject(py)
            }
            Indexer::ForwardSlice(s) => ("ForwardSlice", s).into_pyobject(py),
            Indexer::GeneralSlice(s) => ("GeneralSlice", s).into_pyobject(py),
            Indexer::Positions(s) => ("Positions", s).into_pyobject(py),
            Indexer::Mask(s) => ("Mask", s).into_pyobject(py),
            Indexer::Label(s) => ("Label", s).into_pyobject(py),
            Indexer::Advanced(s) => ("Advanced", s).into_pyobject(py),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_len_matches_forward_and_reverse_slices() {
        let step = |s| NonZeroI64::new(s);
        assert_eq!(slice_len(Some(0), Some(10), step(1), 10), 10);
        assert_eq!(slice_len(Some(1), Some(9), step(2), 10), 4);
        assert_eq!(slice_len(Some(-8), Some(-1), step(2), 10), 4);
        assert_eq!(slice_len(Some(9), Some(0), step(-2), 10), 5);
    }

    #[test]
    fn slice_indexers_report_the_selected_size() {
        let forward = Indexer::ForwardSlice(ForwardSlice {
            start: Some(2),
            stop: Some(9),
            step: NonZeroU64::new(2),
        });
        let general = Indexer::GeneralSlice(GeneralSlice {
            value: Slice {
                start: Some(-8),
                stop: Some(-1),
                step: NonZeroI64::new(2),
            },
        });

        assert_eq!(forward.size(10), Some(4));
        assert_eq!(general.size(10), Some(4));
    }

    /// A missing bound is a sentinel, not `-1`: each of these came back as `Some(0)`
    /// while the defaults were real indices. Expected values are CPython's
    /// `len(range(10)[s])`.
    #[test]
    fn reversed_slices_with_a_missing_bound_match_cpython() {
        let general = |start, stop, step| {
            Indexer::GeneralSlice(GeneralSlice {
                value: Slice { start, stop, step: NonZeroI64::new(step) },
            })
            .size(10)
        };
        assert_eq!(general(None, None, -1), Some(10)); // x[::-1]
        assert_eq!(general(Some(5), None, -1), Some(6)); // x[5::-1]
        assert_eq!(general(None, None, -2), Some(5)); // x[::-2]
        assert_eq!(general(None, Some(5), -1), Some(4)); // x[:5:-1]
    }

    /// The step extremes: `i64::MIN` once overflowed negating, and a `u64` step above
    /// `i64::MAX` once wrapped negative.
    #[test]
    fn extreme_steps_match_cpython() {
        let general = Indexer::GeneralSlice(GeneralSlice {
            value: Slice { start: None, stop: None, step: NonZeroI64::new(i64::MIN) },
        });
        let forward = Indexer::ForwardSlice(ForwardSlice {
            start: None,
            stop: None,
            step: NonZeroU64::new(1 << 63),
        });
        assert_eq!(general.size(10), Some(1)); // x[::-2**63]
        assert_eq!(forward.size(1), Some(1)); // x[::2**63], length 1
    }

    #[test]
    fn collection_and_label_indexers_report_known_sizes() {
        let positions = Indexer::Positions(Positions {
            values: vec![1, 3, 5],
        });
        let mask = Indexer::Mask(Mask {
            values: vec![true, false, true],
        });
        let known_label = Indexer::Label(Label {
            value: PayloadRef::for_test(0),
            known_size: Some(7),
        });
        let unknown_label = Indexer::Label(Label {
            value: PayloadRef::for_test(1),
            known_size: None,
        });

        assert_eq!(positions.size(99), Some(3));
        assert_eq!(mask.size(99), Some(2));
        assert_eq!(known_label.size(99), Some(7));
        assert_eq!(unknown_label.size(99), None);
    }
}

#[cfg(test)]
mod properties {
    use super::*;
    use proptest::prelude::*;

    /// The oracle: CPython's own answer, `len(range(n)[slice(start, stop, step)])` -- the
    /// same `slice.indices` resolution the Python side's `GeneralSlice.size` uses.
    fn cpython_len<T: for<'py> IntoPyObject<'py> + Copy>(
        start: Option<T>,
        stop: Option<T>,
        step: Option<T>,
        n: i64,
    ) -> i64 {
        Python::initialize();
        Python::attach(|py| {
            let builtins = py.import("builtins").unwrap();
            let slice = builtins.getattr("slice").unwrap().call1((start, stop, step)).unwrap();
            let range = builtins.getattr("range").unwrap().call1((n,)).unwrap();
            range.get_item(slice).unwrap().len().unwrap() as i64
        })
    }

    /// Bounds that reach the edges: missing, near the dim, anywhere, and the extremes.
    fn bound() -> impl Strategy<Value = Option<i64>> {
        prop_oneof![
            Just(None),
            (-60i64..60).prop_map(Some),
            any::<i64>().prop_map(Some),
            prop_oneof![Just(i64::MIN), Just(i64::MAX)].prop_map(Some),
        ]
    }

    fn step() -> impl Strategy<Value = Option<NonZeroI64>> {
        prop_oneof![
            Just(None),
            (-5i64..=5).prop_filter_map("zero step", NonZeroI64::new).prop_map(Some),
            any::<NonZeroI64>().prop_map(Some),
            Just(NonZeroI64::new(i64::MIN)),
        ]
    }

    fn forward_bound() -> impl Strategy<Value = Option<u64>> {
        prop_oneof![Just(None), (0u64..60).prop_map(Some), any::<u64>().prop_map(Some)]
    }

    proptest! {
        #[test]
        fn general_slice_size_matches_cpython(
            start in bound(), stop in bound(), step in step(), n in 0i64..50,
        ) {
            let slice = GeneralSlice { value: Slice { start, stop, step } };
            let expected = cpython_len(start, stop, step.map(NonZeroI64::get), n);
            prop_assert_eq!(slice.size(n), Some(expected));
        }

        #[test]
        fn forward_slice_size_matches_cpython(
            start in forward_bound(),
            stop in forward_bound(),
            step in prop_oneof![Just(None), (1u64..5).prop_map(NonZeroU64::new), any::<NonZeroU64>().prop_map(Some)],
            n in 0i64..50,
        ) {
            let slice = ForwardSlice { start, stop, step };
            let expected = cpython_len(start, stop, step.map(NonZeroU64::get), n);
            prop_assert_eq!(slice.size(n), Some(expected));
        }
    }
}
