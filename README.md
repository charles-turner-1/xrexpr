# XREXPR: Xarray Expression Rewriter

Imagine you have an xarray dataset that you want to do some analysis on. You might write something like this:

```python
%%timeit 
ds.mean(dim="lat").mean(dim="lon").isel(time=0).compute()
```
`193 ms ± 49.6 ms per loop (mean ± std. dev. of 5 runs, 5 loops each)`


However, it would be a lot faster if you instead wrote:

```python
ds.isel(time=0).mean(dim="lat").mean(dim="lon").compute()
```
`925 μs ± 401 μs per loop (mean ± std. dev. of 5 runs, 5 loops each)`

In this instance, just reordering the operations makes a ~200x performance difference. We can see that these two expressions are equivalent, but unfortunately, xarray can't automatically reorder them for us (yet?). 

```python
from xarray.testing import assert_equal
assert_equal(
    ds.isel(time=0).mean(dim="lat").mean(dim="lon"),
    ds.mean(dim="lat").mean(dim="lon").isel(time=0),
)

# Does not raise an AssertionError
```

That's where `xrexpr` comes in. It takes a function of the form
```python
def func(ds: xr.Dataset) -> xr.Dataset:
    return ds.operation1().operation2()...
```

and reorders the operations (hopefully safely :fingers_crossed:) to optimize the performance of the expression.

```python

>>> from xrexpr import peek_rewritten_expr, rewrite_expr

>>> def slow_func(ds: xr.Dataset) -> xr.Dataset:
        return ds.mean(dim="lat").mean(dim="lon").isel(time=0)

>>> peek_rewritten_expr(func)
"""
def func(ds: xr.Dataset) -> xr.Dataset:
    return ds.isel(time=0).mean(dim="lat").mean(dim="lon")
"""
```

```python
%%timeit
func(ds)
```
`925 μs ± 401 μs per loop (mean ± std. dev. of 5 runs, 5 loops each)`

```python
%%timeit
rewritten_func = rewrite_expr(slow_func)
rewritten_func(ds)
```
`2.43 ms ± 546 μs per loop (mean ± std. dev. of 5 runs, 5 loops each)`

(Note that in the above example, we are also timing the rewriting process itself. We could do that separately once, in which case the performance would be even better - aroun the 900µs for the fast case.)

```python
rewritten_func(ds)
```
`795 μs ± 299 μs per loop (mean ± std. dev. of 5 runs, 5 loops each)`

That's it! Now you can use `func` as you normally would, and it will automatically reorder the operations for you to optimize performance.



___

This package is just making it's way out of the proof of concept stage, so expect some issues. It is also unlikely to support the full range of xarray operations for some time. If it doesn't do anything for you, please open an issue!
