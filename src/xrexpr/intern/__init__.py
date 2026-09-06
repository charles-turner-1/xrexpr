"""Interning: relabel an IR plan's *names* to handles for a Rust backend, structure intact.

``interner`` holds the process-wide :class:`Interner` singleton and the :class:`InternedVal`
handle newtype; ``ir`` holds the ``Interned*`` record types (the interned counterpart of
:mod:`xrexpr.ir`); ``converters`` holds the selective :func:`intern`/:func:`deintern` pair.
This package re-exports the public surface so callers need only ``xrexpr.intern``.
"""

from xrexpr.intern.converters import deintern, intern
from xrexpr.intern.interner import InternedVal, Interner
from xrexpr.intern.ir import (
    InternedContextOpen,
    InternedDrop,
    InternedElementwise,
    InternedFluentOp,
    InternedGroupedReduce,
    InternedLoweredOp,
    InternedOp,
    InternedOpaque,
    InternedProject,
    InternedRechunk,
    InternedReduce,
    InternedRename,
    InternedScan,
    InternedSelect,
    InternedWeightedReduce,
    InternedWindowedReduce,
)

__all__ = [
    "InternedContextOpen",
    "InternedDrop",
    "InternedElementwise",
    "InternedFluentOp",
    "InternedGroupedReduce",
    "InternedLoweredOp",
    "InternedOp",
    "InternedOpaque",
    "InternedProject",
    "InternedRechunk",
    "InternedReduce",
    "InternedRename",
    "InternedScan",
    "InternedSelect",
    "InternedVal",
    "InternedWeightedReduce",
    "InternedWindowedReduce",
    "Interner",
    "deintern",
    "intern",
]
