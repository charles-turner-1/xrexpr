from typing import Any

from xrexpr.intern.ir import InternedFluentOp, InternedLoweredOp

# Rust's view of an op: ``(record class name, {reasoning field: value})``. See
# ``rust/src/interned_ir.rs``.
_OpView = tuple[str, dict[str, Any]]

def view_fluent_plan(ops: list[InternedFluentOp]) -> list[_OpView]: ...
def view_lowered_plan(ops: list[InternedLoweredOp]) -> list[_OpView]: ...
