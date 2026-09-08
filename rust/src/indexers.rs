use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyFrozenSet, PyTuple};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};



pub struct Scalar{
    value: InternedVal,
    drops_dims: bool,
    position: Option<InternedVal>
}

pub enum Indexer {
    Scalar,
    ForwardSlice,
    GeneralSlice,
    Positions,
    Mask,
    Label,
    Advanced
}

