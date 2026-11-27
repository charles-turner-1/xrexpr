//! The interned IR as Rust sees it: one struct per `xrexpr.intern.ir.Interned*` record,
//! holding the record's *reasoning* fields -- every field not marked `Payload` on the Python
//! side. The `Payload` replay header (`args`/`kwargs`) is not extracted.
//!
//! Each struct derives `FromPyObject` (Python record -> Rust, reading each field by
//! attribute name) and `IntoPyObject` (Rust -> a Python dict of the same fields). The dict
//! is Rust's *view* of the record, not the record rebuilt: the boundary property test
//! compares it with the same projection made in Python, which is how a field Rust misreads
//! or misses gets caught.

use pyo3::conversion::FromPyObjectOwned;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyTuple};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use crate::indexers::Indexer;
use crate::chunks::ChunkSpec;
use crate::ir::{AllDims, ALL_DIMS};

/// A dimension in a dataset, interned from (typically) a string to an int.
/// For example, "time", "lat", "lon", etc. > 0, 1, 2, etc.
///
/// Deriving `IntoPyObject` on a one-field tuple struct makes it *transparent*: it converts
/// to its inner value, so a handle goes back to Python as a plain `int`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, IntoPyObject)]
pub struct InternedVal(pub i32);

impl<'py> FromPyObject<'_, '_> for InternedVal {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> PyResult<Self> {
        Ok(InternedVal(obj.getattr("handle")?.extract::<i32>()?))
    }
}

/// Read any Python mapping into a `HashMap`, through its `items()`.
///
/// pyo3's own `HashMap` extraction accepts only a real `dict`. The records hold
/// `frozendict`s, and whether a `frozendict` *is* a `dict` depends on how it was built: the
/// pure-Python package subclasses `dict`, its C extension (conda-forge's Python 3.10 build)
/// does not. Every mapping field goes through here so neither build matters.
///
/// Used as `#[pyo3(from_py_with = extract_mapping)]`; the compiler infers `K` and `V` from
/// the field's type.
fn extract_mapping<'py, K, V>(obj: &Bound<'py, PyAny>) -> PyResult<HashMap<K, V>>
where
    K: FromPyObjectOwned<'py> + Eq + Hash,
    V: FromPyObjectOwned<'py>,
{
    obj.call_method0("items")?
        .try_iter()?
        .map(|item| item?.extract::<(K, V)>())
        .collect()
}

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum DimSet {
    AllDims,
    Concrete(std::collections::HashSet<InternedVal>),
}

impl FromPyObject<'_, '_> for DimSet {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> Result<Self, Self::Error> {
        if obj.is_instance_of::<AllDims>() {
            Ok(DimSet::AllDims)
        } else {
            Ok(DimSet::Concrete(obj.extract()?))
        }
    }
}

/// `AllDims` goes back as the `ALL_DIMS` singleton, a concrete set as a `frozenset` of
/// handles -- the same shapes `xrexpr.intern.ir.DimSet` uses.
impl<'py> IntoPyObject<'py> for DimSet {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            DimSet::AllDims => Ok(ALL_DIMS.import(py, "xrexpr._xrexprs.ir", "ALL_DIMS")?.clone()),
            DimSet::Concrete(dims) => Ok(PyFrozenSet::new(py, dims)?.into_any()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedReduce{
    name: String,
    consumes: DimSet,
    keepdims: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedSelect{
    name: String,
    #[pyo3(from_py_with = extract_mapping)]
    indexer: HashMap<InternedVal, Indexer>
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedScan{
    name: String,
    dims: DimSet,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedElementwise{
    name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedProject{
    name: String,
    variables: Vec<InternedVal>,
    single: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedRechunk{
    name: String,
    #[pyo3(from_py_with = extract_mapping)]
    chunks: HashMap<InternedVal, ChunkSpec>,
    /// The one spec every dim gets (`chunk(10)`), or `None` for a per-dim mapping.
    uniform: Option<ChunkSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedOpaque{
    name: String
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedDrop{
    name: String,
    variables: Vec<InternedVal>,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedRename{
    name: String,
    #[pyo3(from_py_with = extract_mapping)]
    mapping: HashMap<InternedVal, InternedVal>,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedContextOpen{
    name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedGroupedReduce{
    name: String,
    group_dim: InternedVal,
    new_dim: InternedVal,
    reduce: String,
    consumes: HashSet<InternedVal>,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedWindowedReduce{
    name: String,
    reduce: String,
    /// Window length per dim: `rolling(time=3)` is `{time: 3}`.
    #[pyo3(from_py_with = extract_mapping)]
    window: HashMap<InternedVal, i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedWeightedReduce{
    name: String,
    reduce: String,
    weight_dims: HashSet<InternedVal>,
    consumes: DimSet
}

/// Defines an enum whose variants each wrap one interned record, plus its two conversions:
///
/// - `FromPyObject` dispatches on the Python object's *class name*. pyo3's derived enum
///   extraction would instead try each variant in turn and take the first whose fields
///   happen to be readable, which can silently pick the wrong one -- every record has a
///   `name`, and `name` is all an `InternedElementwise` reads, so an `InternedProject` would
///   come out as an `Elementwise`.
/// - `IntoPyObject` gives back `(class name, fields dict)`, so the variant survives the trip.
///
/// The class name is the wrapped type's own name (`stringify!(InternedReduce)` is
/// `"InternedReduce"`), which is why every struct here is named after its Python record.
macro_rules! op_enum {
    ($(#[$meta:meta])* $enum:ident { $($variant:ident($record:ident)),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum $enum {
            $($variant($record)),+
        }

        impl FromPyObject<'_, '_> for $enum {
            type Error = PyErr;
            fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> PyResult<Self> {
                let ty = obj.get_type().name()?;
                match ty.to_str()? {
                    $(stringify!($record) => Ok($enum::$variant(obj.extract()?)),)+
                    other => Err(PyTypeError::new_err(format!(
                        "cannot convert {} to {}", other, stringify!($enum)
                    ))),
                }
            }
        }

        impl<'py> IntoPyObject<'py> for $enum {
            type Target = PyTuple;
            type Output = Bound<'py, PyTuple>;
            type Error = PyErr;

            fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
                match self {
                    $($enum::$variant(record) => (stringify!($record), record).into_pyobject(py),)+
                }
            }
        }
    };
}

op_enum! {
    /// An op in a fluent plan: `xrexpr.intern.ir.InternedFluentOp`.
    FluentOp {
        Reduce(InternedReduce),
        Select(InternedSelect),
        Scan(InternedScan),
        Elementwise(InternedElementwise),
        Project(InternedProject),
        Rechunk(InternedRechunk),
        Opaque(InternedOpaque),
        Drop(InternedDrop),
        Rename(InternedRename),
        ContextOpen(InternedContextOpen),
    }
}

op_enum! {
    /// An op in a lowered plan: `xrexpr.intern.ir.InternedLoweredOp`.
    LoweredOp {
        Reduce(InternedReduce),
        Select(InternedSelect),
        Scan(InternedScan),
        Elementwise(InternedElementwise),
        Project(InternedProject),
        Rechunk(InternedRechunk),
        Opaque(InternedOpaque),
        Drop(InternedDrop),
        Rename(InternedRename),
        GroupedReduce(InternedGroupedReduce),
        WindowedReduce(InternedWindowedReduce),
        WeightedReduce(InternedWeightedReduce),
    }
}

/// Extract an interned fluent plan into Rust and hand back Rust's view of each op, as
/// `(class name, fields)` tuples. The plan crosses the boundary in both directions and
/// nothing else happens to it -- it exists so the boundary itself can be tested.
#[pyfunction]
pub fn view_fluent_plan(ops: Vec<FluentOp>) -> Vec<FluentOp> {
    ops
}

/// The lowered-plan counterpart of [`view_fluent_plan`].
#[pyfunction]
pub fn view_lowered_plan(ops: Vec<LoweredOp>) -> Vec<LoweredOp> {
    ops
}
