use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyFrozenSet, PyTuple};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

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

static ALL_DIMS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// A dimension in a dataset, represented as a string. For example, "time", "lat", "lon", etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
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

#[derive(Clone, PartialEq)]
pub enum DimSet {
    AllDims,
    Concrete(std::collections::HashSet<Dim>),
}

impl Hash for DimSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            DimSet::AllDims => {
                0.hash(state);
            }
            DimSet::Concrete(dim_set) => {
                let mut dims = dim_set.iter().collect::<Vec<_>>();
                dims.sort(); // Can't do this above or we get a unit ouput
                dims.hash(state)
            }
        }
    }
}

impl FromPyObject<'_, '_> for DimSet {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'_, '_, pyo3::PyAny>) -> Result<Self, Self::Error> {
        if obj.is_instance_of::<AllDims>() {
            Ok(DimSet::AllDims)
        } else {
            let dims: HashSet<Dim> = obj
                .to_owned()
                .try_iter()?
                .map(|item| {
                    let dim_str: String = item?.extract()?;
                    Ok(Dim(dim_str))
                })
                .collect::<PyResult<_>>()?;
            Ok(DimSet::Concrete(dims))
        }
    }
}

impl<'py> IntoPyObject<'py> for DimSet {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            DimSet::AllDims => Ok(ALL_DIMS
                .import(py, "xrexpr._xrexprs.ir", "ALL_DIMS")?
                .clone()),
            DimSet::Concrete(dim_set) => {
                let dim_list: Vec<String> = dim_set.into_iter().map(|d| d.0).collect();
                Ok(PyFrozenSet::new(py, dim_list)?.into_any())
            }
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
    #[pyo3(get)]
    name: String,
    /// The call's positional arguments, verbatim.
    /// Can't just use a pyo3(get) here, as we need it to come back as a tuple,
    /// not a list (default behaviour)
    args: Vec<Py<PyAny>>,
    /// The call's keyword arguments, verbatim.
    #[pyo3(get)]
    kwargs: std::collections::HashMap<String, Py<PyAny>>,
    /// The dims the reduction consumes. Typically named in the call, but if none
    /// are named, then it consumes all dims (``ALL_DIMS``).
    #[pyo3(get)]
    consumes: DimSet,
    /// Whether the reduction keeps its named dims at size 1 (``keepdims=True``).
    #[pyo3(get)]
    keepdims: bool,
}

#[pymethods]
impl Reduce {
    #[new]
    #[pyo3(signature=(name, args=Vec::new(), kwargs=HashMap::new(), consumes=DimSet::Concrete(HashSet::new())))]
    pub fn new(
        py: Python,
        name: String,
        args: Vec<Py<PyAny>>,
        kwargs: std::collections::HashMap<String, Py<PyAny>>,
        consumes: DimSet,
    ) -> Self {
        let keepdims = kwargs
            .get("keepdims")
            .is_none_or(|v| v.extract::<bool>(py).unwrap_or(false));
        Reduce {
            name,
            args,
            kwargs,
            consumes,
            keepdims,
        }
    }

    #[getter]
    fn args<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.args)
    }

    fn __hash__(&self, py: Python) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        self.keepdims.hash(&mut hasher);
        self.consumes.hash(&mut hasher);
        // Just gotta add args & kwargs now.
        self.hash_args(py, &mut hasher).unwrap();
        self.hash_kwargs(py, &mut hasher).unwrap();
        hasher.finish()
    }

    /// Fast check for rust fields - easy. Then compare hashes. If everything
    /// matches, then we need to check args and kwargs for equality, because
    /// hashes might collide. See python -c "hash(-1) == hash(-2)"
    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python) -> PyResult<bool> {
        // Fast check rust fields first
        let Ok(other) = other.cast::<Reduce>() else {
            return Ok(false);
        };
        let other = other.borrow();
        // let py = other.py(); // Can do this to remove the py argment.

        if self.name != other.name {
            return Ok(false);
        }
        if self.keepdims != other.keepdims {
            return Ok(false);
        }
        if self.consumes != other.consumes {
            return Ok(false);
        }

        // Now, we need to check args and kwargs for equality, because hashes might collide.
        // args
        if self.args.len() != other.args.len() {
            return Ok(false);
        }
        for (s_arg, o_arg) in self.args.iter().zip(other.args.iter()) {
            if !s_arg.bind(py).eq(o_arg.bind(py))? {
                return Ok(false);
            }
        }
        // kwargs
        if self.kwargs.len() != other.kwargs.len() {
            return Ok(false);
        }
        for (s_key, s_val) in self.kwargs.iter() {
            let Some(o_val) = other.kwargs.get(s_key) else {
                return Ok(false);
            };
            if !s_val.bind(py).eq(o_val.bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Reduce {
    fn hash_args(&self, py: Python, state: &mut impl Hasher) -> PyResult<()> {
        // Call __hash__ on args from Python side so we can fold them into our
        // rust hash.
        for arg in &self.args {
            let arg_hash = arg.bind(py).hash()?;
            arg_hash.hash(state);
        }
        Ok(())
    }

    fn hash_kwargs(&self, py: Python, state: &mut impl Hasher) -> PyResult<()> {
        let mut kwargs: Vec<_> = self.kwargs.iter().collect();
        kwargs.sort_by_key(|k| k.0);

        // Now iterate over the sorted keys and hash the key-value pairs.
        for (key, value) in kwargs {
            key.hash(state);
            let value_hash = value.bind(py).hash()?;
            value_hash.hash(state);
        }
        Ok(())
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
