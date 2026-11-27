//! The interned IR in Rust: one struct per `xrexpr.intern.ir.Interned*` record, with every
//! field of the Python record.
//!
//! Semantic fields (dims, indexers, chunk specs, flags) are real Rust data that rules can
//! reason about. Payload fields (the replay header: `args`/`kwargs`, and a fused node's
//! `reduce_args`/`reduce_kwargs`) are [`PayloadRef`]s: Python swaps each one for a handle
//! before the plan crosses (`xrexpr.intern.crossing.to_rust`), and Rust can only carry it.
//!
//! Each struct derives `FromPyObject` (Python record -> Rust, reading each field by
//! attribute name) and `IntoPyObject` (Rust -> a dict of the same fields). The enums below
//! return `(class name, fields)`, and `xrexpr.intern.crossing.from_rust` rebuilds the
//! Python record from that. Nothing is dropped, so a plan comes back exactly as it went in
//! unless a rule changed it.

use pyo3::conversion::FromPyObjectOwned;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyFrozenSet, PyTuple, PyType};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use crate::indexers::Indexer;
use crate::chunks::ChunkSpec;
use crate::ir::{AllDims, ALL_DIMS};
use crate::payload::PayloadRef;

/// A dimension or variable name, interned from (typically) a string to an int.
/// For example, "time", "lat", "lon" become 0, 1, 2.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct InternedVal(pub i32);

/// Python's `xrexpr.intern.interner.InternedVal` class, imported once.
static INTERNED_VAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Converts back to Python's `InternedVal`, not a plain `int`, so the Python rebuild can
/// tell a name from a number (a position, a size).
impl<'py> IntoPyObject<'py> for InternedVal {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        INTERNED_VAL
            .import(py, "xrexpr.intern.interner", "InternedVal")?
            .call1((self.0,))
    }
}

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
/// handles: the same shapes `xrexpr.intern.ir.DimSet` uses.
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
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedSelect{
    name: String,
    #[pyo3(from_py_with = extract_mapping)]
    indexer: HashMap<InternedVal, Indexer>,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedScan{
    name: String,
    dims: DimSet,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedElementwise{
    name: String,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedProject{
    name: String,
    variables: Vec<InternedVal>,
    single: bool,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedRechunk{
    name: String,
    #[pyo3(from_py_with = extract_mapping)]
    chunks: HashMap<InternedVal, ChunkSpec>,
    /// The one spec every dim gets (`chunk(10)`), or `None` for a per-dim mapping.
    uniform: Option<ChunkSpec>,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedOpaque{
    name: String,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedDrop{
    name: String,
    variables: Vec<InternedVal>,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedRename{
    name: String,
    #[pyo3(from_py_with = extract_mapping)]
    mapping: HashMap<InternedVal, InternedVal>,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedContextOpen{
    name: String,
    args: PayloadRef,
    kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedGroupedReduce{
    name: String,
    group_dim: InternedVal,
    new_dim: InternedVal,
    reduce: String,
    consumes: HashSet<InternedVal>,
    args: PayloadRef,
    kwargs: PayloadRef,
    reduce_args: PayloadRef,
    reduce_kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedWindowedReduce{
    name: String,
    reduce: String,
    /// Window length per dim: `rolling(time=3)` is `{time: 3}`.
    #[pyo3(from_py_with = extract_mapping)]
    window: HashMap<InternedVal, i64>,
    args: PayloadRef,
    kwargs: PayloadRef,
    reduce_args: PayloadRef,
    reduce_kwargs: PayloadRef,
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub struct InternedWeightedReduce{
    name: String,
    reduce: String,
    weight_dims: HashSet<InternedVal>,
    consumes: DimSet,
    args: PayloadRef,
    kwargs: PayloadRef,
    reduce_args: PayloadRef,
    reduce_kwargs: PayloadRef,
}

/// Defines an enum whose variants each wrap one interned record, plus its two conversions:
///
/// - `FromPyObject` dispatches on the Python object's *class name*. pyo3's derived enum
///   extraction would instead try each variant in turn and take the first whose fields
///   happen to be readable, which can silently pick the wrong one -- every record has a
///   `name`, and `name` is all an `InternedElementwise` reads, so an `InternedProject` would
///   come out as an `Elementwise`.
/// - `IntoPyObject` gives back `(class name, fields dict)`, so the variant survives the trip
///   and Python can rebuild the record.
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

/// Extract a crossed fluent plan into Rust and hand each op straight back, as
/// `(class name, fields)`. Nothing happens to the plan in between: this exists so the
/// boundary can be tested (Python checks the rebuilt plan equals the one it sent).
#[pyfunction]
pub fn roundtrip_fluent_plan(ops: Vec<FluentOp>) -> Vec<FluentOp> {
    ops
}

/// The lowered-plan counterpart of [`roundtrip_fluent_plan`].
#[pyfunction]
pub fn roundtrip_lowered_plan(ops: Vec<LoweredOp>) -> Vec<LoweredOp> {
    ops
}
