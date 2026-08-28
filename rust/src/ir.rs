use pyo3::prelude::*;
use std::fmt::{Display, Formatter};

// AllDims just represents every dimension in a dataset. Has no real fields
#[derive(Debug, Clone, Copy)]
#[pyclass(module="xrexpr._xrexprs.ir",skip_from_py_object)]
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
