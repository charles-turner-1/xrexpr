//! `PayloadRef`: a handle to a Python object that Rust carries but can never look inside.
//!
//! Some fields of a plan are *payloads*: the verbatim `args`/`kwargs` of a call, the labels
//! of a `sel`, a chunk spec nothing models. Python needs them back exactly as they were to
//! replay the plan, and no rule has any business reading them. Before a plan crosses into
//! Rust, Python swaps each payload for a handle into a per-crossing table
//! (`xrexpr.intern.crossing.PayloadTable`), and swaps it back afterwards.
//!
//! The handle's field is private, and there's no way to make one except by converting from
//! Python's `xrexpr.payload.PayloadRef`. So Rust code can copy, compare and return a
//! `PayloadRef`, but can't create one from a number, read its index, or do arithmetic on
//! it. If a rule ever needs something that's inside a payload, that thing has to become a
//! proper semantic field instead (the way `keepdims` is).

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

/// A handle to one payload in the crossing's table. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PayloadRef(u32);

impl PayloadRef {
    /// Make a ref directly, for Rust unit tests that build records by hand.
    #[cfg(test)]
    pub(crate) fn for_test(index: u32) -> Self {
        PayloadRef(index)
    }
}

/// Python's `xrexpr.payload.PayloadRef` class, imported once.
static PAYLOAD_REF: PyOnceLock<Py<PyType>> = PyOnceLock::new();

impl FromPyObject<'_, '_> for PayloadRef {
    type Error = PyErr;

    /// Read `PayloadRef.index`. Anything that isn't a Python `PayloadRef` is rejected, so
    /// a payload that wasn't swapped out (a raw tuple, say) fails loudly instead of being
    /// read as something else.
    fn extract(obj: pyo3::Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let cls = PAYLOAD_REF.import(obj.py(), "xrexpr.payload", "PayloadRef")?;
        if !obj.is_instance(cls)? {
            return Err(PyTypeError::new_err(format!(
                "expected a PayloadRef, got {}",
                obj.get_type().name()?
            )));
        }
        Ok(PayloadRef(obj.getattr("index")?.extract()?))
    }
}

/// Converts back to Python's `PayloadRef`, so Python can tell a payload from a plain int.
impl<'py> IntoPyObject<'py> for PayloadRef {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        PAYLOAD_REF
            .import(py, "xrexpr.payload", "PayloadRef")?
            .call1((self.0,))
    }
}
