use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::fmt::{Display, Formatter};

/// A singleton type representing all dimensions in a dataset.
/// This is used to indicate that an operation should be applied to all dimensions,
/// rather than a specific subset of dimensions.
/// Handy for things like `ds.mean()` which are implicitly over all dimensions.
#[derive(Debug, Clone, Copy)]
#[pyclass(module = "xrexpr._xrexprs.ir", skip_from_py_object)]
struct AllDims;

impl Display for AllDims {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ALL_DIMS")
    }
}

#[pymethods]
impl AllDims {
    #[new]
    fn new() -> Self {
        AllDims
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    fn __eq__(&self, other: &Bound<PyAny>) -> bool {
        other.is_instance_of::<AllDims>()
    }

    fn __hash__(&self) -> u64 {
        0 // AllDims is a singleton, so we can return a constant hash value
    }
}

/// A dimension in a dataset, represented as a string. For example, "time", "lat", "lon", etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dim(pub String);

impl Display for Dim {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Dim {
    fn from(s: &str) -> Self {
        Dim(s.to_string())
    }
}

pub enum DimSet {
    AllDims,
    Concrete(std::collections::HashSet<Dim>),
}

impl FromPyObject<'_, '_> for DimSet {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> Result<Self, Self::Error> {
        if obj.is_instance_of::<AllDims>() {
            Ok(DimSet::AllDims)
        } else if let Ok(dim_list) = obj.extract::<Vec<String>>() {
            let dim_set: std::collections::HashSet<Dim> =
                dim_list.into_iter().map(|s| Dim(s)).collect();
            Ok(DimSet::Concrete(dim_set))
        } else {
            Err(PyTypeError::new_err(
                "DimSet must be AllDims or a list of strings",
            ))
        }
    }
}

/// A dimension-destroying reduction (``mean``/``sum``/``std``/...).
/// args and kwargs have to be Py<PyAny>, because they can be any Python object,
/// eg. a string, a number, a list, a dict, etc. We don't want to restrict the types of the arguments.
/// We might be able to lock this down in future. We can't use Bound<'_, PyAny>,
/// because otherwise we tie the lifetime of the Reduce struct to the lifetime of
/// the args/kwargs in the Python interpreter, which might mean they get dropped
/// when we still need them here (ie. this wouldn't compile)
#[pyclass(module = "xrexpr._xrexprs.ir", skip_from_py_object)]
pub struct Reduce {
    /// What reduction method we are applying, e.g. "mean", "sum", "std", etc.
    name: String,
    /// The call's positional arguments, verbatim.
    args: Vec<Py<PyAny>>,
    /// The call's keyword arguments, verbatim.
    kwargs: std::collections::HashMap<String, Py<PyAny>>,
    /// The dims the reduction consumes. Typically named in the call, but if none
    /// are named, then it consumes all dims (``ALL_DIMS``).
    consumes: DimSet,
    /// Whether the reduction keeps its named dims at size 1 (``keepdims=True``).
    keepdims: bool,
}

#[pymethods]
impl Reduce {
    #[new]
    pub fn new(
        py: Python,
        name: String,
        args: Vec<Py<PyAny>>,
        kwargs: std::collections::HashMap<String, Py<PyAny>>,
        consumes: DimSet,
    ) -> Self {
        let keepdims = kwargs
            .get("keepdims")
            .map_or(false, |v| v.extract::<bool>(py).unwrap_or(false));
        Reduce {
            name,
            args,
            kwargs,
            consumes,
            keepdims,
        }
    }
}

#[pymodule]
pub mod ir {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::AllDims;
    #[pymodule_export]
    use super::Reduce;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("ALL_DIMS", AllDims)?;

        Ok(())
    }
}
