use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use std::fmt::{Display, Formatter};

/// A singleton type representing all dimensions in a dataset.
/// This is used to indicate that an operation should be applied to all dimensions,
/// rather than a specific subset of dimensions.
/// Handy for things like `ds.mean()` which are implicitly over all dimensions.
#[derive(Debug, Clone, Copy)]
#[pyclass(module = "xrexpr._xrexprs.ir", skip_from_py_object)]
pub struct AllDims;

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

/// The module's `ALL_DIMS` instance, imported once and cached, so Rust hands back the
/// same object Python compares against.
pub(crate) static ALL_DIMS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[pymodule]
pub mod ir {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::AllDims;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("ALL_DIMS", AllDims)?;

        Ok(())
    }
}
