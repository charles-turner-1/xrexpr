import inspect
from collections.abc import Callable
from typing import Any

import libcst as cst
import xarray as xr

from .cst import SelectionPushdown

# TODO: Make tehse actually work nicely as decorators


def rewrite_expr(
    func: Callable[[xr.Dataset], xr.Dataset],
) -> Callable[[xr.Dataset], xr.Dataset]:
    """
    Take a function that operates on an xarray Dataset and rewrite its
    expression tree to push down selection calls (isel, sel) to the first
    position in a chain of mean calls.
    """
    func_source = inspect.getsource(func)
    cst_for_mods = cst.parse_module(func_source)
    transformer = SelectionPushdown()
    transformed_cst = cst_for_mods.visit(transformer)

    namespace: dict[str, Any] = {}
    exec(transformed_cst.code, func.__globals__, namespace)

    return namespace[func.__name__]  # type: ignore[no-any-return]


def peek_rewritten_expr(
    func: Callable[[xr.Dataset], xr.Dataset],
) -> str:
    """
    Return the rewritten expression of a function that operates on an xarray
    Dataset, without executing it.
    """
    func_source = inspect.getsource(func)
    cst_for_mods = cst.parse_module(func_source)
    transformer = SelectionPushdown()
    transformed_func: str = cst_for_mods.visit(transformer).code

    return transformed_func
