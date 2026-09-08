use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyFrozenSet, PyTuple};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

mod indexers;

#[derive(Debug, Clone, Copy)]
#[pyclass(module = "xrexpr._xrexprs.ir", skip_from_py_object)]
struct AllDims;

/// A dimension in a dataset, interned from (typically) a string to an int.
/// For example, "time", "lat", "lon", etc. > 0, 1, 2, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct InternedVal(pub i32);

#[derive(Clone, PartialEq)]
pub enum DimSet {
    AllDims,
    Concrete(std::collections::HashSet<InternedVal>),
}

pub struct InternedReduce{
    name: String,
    comsumes: HashSet<InternedVal>,
    keepdims: bool,
}

pub struct InternedSelect{
    name: String,
    indexer: HashMap<InternedVal, Indexer>
}

pub struct InternedScan{
    name: String,
    dims: DimSet,
}

pub struct InternedElementwise{
    name: String,
}

pub struct InternetProject{
    name: String,
    variables: Vec<InternedVal>,
    single: bool,
}

pub struct InternedRechunk{
    name: String,
    chunks: HashMap<InternedVal, ChunkSpec>,
}

pub struct InternedOpaque{
    name: String
}

pub struct InternedDrop{
    name: String,
    variables: Vec<InternedVal>,
}

pub struct InternedRename{
    name: String,
    mapping: HashMap<InternedVal, InternedVal>,
}

pub struct InternedContextOpen{
    name: String,
}

pub struct InternedGroupedReduce{
    name: String,
    group_dim: InternedVal,
    new_dim: InternedVal,
    reduce: String,
    consumes: HashSet<InternedVal>,
}

pub struct InternedWindowedReduce{
    name: String,
    reduce: String,
    window: String,
}
 pub struct InternedWeightedReduce{
    name: String,
    reduce: String,
    weight_dims: HashSet<InternedVal>,
    consumes: DimSet
}
