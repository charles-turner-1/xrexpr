use pyo3::prelude::*;

mod ir;
/// A Python module implemented in Rust.
#[pymodule]
mod _xrexprs {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::ir::ir;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        let sys = py.import("sys")?;
        let modules = sys.getattr("modules")?;
        let ir = m.getattr("ir")?;
        modules.set_item("xrexpr._xrexprs.ir", ir)?; // now the dotted import resolves
        Ok(())
    }
}
