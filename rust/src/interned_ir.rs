use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, };
use crate::indexers::Indexer;
use crate::chunks::ChunkSpec;
use crate::ir::{AllDims};

/// A dimension in a dataset, interned from (typically) a string to an int.
/// For example, "time", "lat", "lon", etc. > 0, 1, 2, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct InternedVal(pub i32);

impl<'py> FromPyObject<'_, '_> for InternedVal {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> PyResult<Self> {
        Ok(InternedVal(obj.getattr("handle")?.extract::<i32>()?))
    }
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

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedReduce{
    name: String,
    consumes: DimSet,
    keepdims: bool,
}

impl From<InternedReduce> for FluentOp {
    fn from(r: InternedReduce) -> Self {
        FluentOp::Reduce(r)
    }
}

impl From<InternedReduce> for LoweredOp {
    fn from(r: InternedReduce) -> Self {
        LoweredOp::Reduce(r)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedSelect{
    name: String,
    indexer: HashMap<InternedVal, Indexer>
}

impl From<InternedSelect> for FluentOp {
    fn from(s: InternedSelect) -> Self {
        FluentOp::Select(s)
    }
}

impl From<InternedSelect> for LoweredOp {
    fn from(s: InternedSelect) -> Self {
        LoweredOp::Select(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedScan{
    name: String,
    dims: DimSet,
}

impl From<InternedScan> for FluentOp {
    fn from(s: InternedScan) -> Self {
        FluentOp::Scan(s)
    }
}

impl From<InternedScan> for LoweredOp {
    fn from(s: InternedScan) -> Self {
        LoweredOp::Scan(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedElementwise{
    name: String,
}

impl From<InternedElementwise> for FluentOp {
    fn from(s: InternedElementwise) -> Self {
        FluentOp::Elementwise(s)
    }
}

impl From<InternedElementwise> for LoweredOp {
    fn from(s: InternedElementwise) -> Self {
        LoweredOp::Elementwise(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedProject{
    name: String,
    variables: Vec<InternedVal>,
    single: bool,
}

impl From<InternedProject> for FluentOp {
    fn from(p: InternedProject) -> Self {
        FluentOp::Project(p)
    }
}

impl From<InternedProject> for LoweredOp {
    fn from(p: InternedProject) -> Self {
        LoweredOp::Project(p)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedRechunk{
    name: String,
    chunks: HashMap<InternedVal, ChunkSpec>,
}

impl From<InternedRechunk> for FluentOp {
    fn from(r: InternedRechunk) -> Self {
        FluentOp::Rechunk(r)
    }
}

impl From<InternedRechunk> for LoweredOp {
    fn from(r: InternedRechunk) -> Self {
        LoweredOp::Rechunk(r)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedOpaque{
    name: String
}

impl From<InternedOpaque> for FluentOp {
    fn from(o: InternedOpaque) -> Self {
        FluentOp::Opaque(o)
    }
}

impl From<InternedOpaque> for LoweredOp {
    fn from(o: InternedOpaque) -> Self {
        LoweredOp::Opaque(o)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedDrop{
    name: String,
    variables: Vec<InternedVal>,
}

impl From<InternedDrop> for FluentOp {
    fn from(d: InternedDrop) -> Self {
        FluentOp::Drop(d)
    }
}

impl From<InternedDrop> for LoweredOp {
    fn from(d: InternedDrop) -> Self {
        LoweredOp::Drop(d)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedRename{
    name: String,
    mapping: HashMap<InternedVal, InternedVal>,
}

impl From<InternedRename> for FluentOp {
    fn from(r: InternedRename) -> Self {
        FluentOp::Rename(r)
    }
}

impl From<InternedRename> for LoweredOp {
    fn from(r: InternedRename) -> Self {
        LoweredOp::Rename(r)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedContextOpen{
    name: String,
}

impl From<InternedContextOpen> for FluentOp {
    fn from(c: InternedContextOpen) -> Self {
        FluentOp::ContextOpen(c)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedGroupedReduce{
    name: String,
    group_dim: InternedVal,
    new_dim: InternedVal,
    reduce: String,
    consumes: HashSet<InternedVal>,
}

impl From<InternedGroupedReduce> for LoweredOp {
    fn from(g: InternedGroupedReduce) -> Self {
        LoweredOp::GroupedReduce(g)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedWindowedReduce{
    name: String,
    reduce: String,
    window: String,
}

impl From<InternedWindowedReduce> for LoweredOp {
    fn from(w: InternedWindowedReduce) -> Self {
        LoweredOp::WindowedReduce(w)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromPyObject)]
pub struct InternedWeightedReduce{
    name: String,
    reduce: String,
    weight_dims: HashSet<InternedVal>,
    consumes: DimSet
}

impl From<InternedWeightedReduce> for LoweredOp {
    fn from(w: InternedWeightedReduce) -> Self{
        LoweredOp::WeightedReduce(w)
    }
}


// Not really used anywhere, just defined to keep the common ops of fluent and
// lowered listed.
pub enum InternedOp {
    Reduce(InternedReduce),
    Select(InternedSelect),
    Scan(InternedScan),
    Elementwise(InternedElementwise),
    Project(InternedProject),
    Rechunk(InternedRechunk),
    Opaque(InternedOpaque),
    Drop(InternedDrop),
    Rename(InternedRename),
}

pub enum FluentOp{
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

pub enum LoweredOp {
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
