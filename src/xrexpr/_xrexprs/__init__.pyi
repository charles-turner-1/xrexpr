from typing import Any

# Rust hands each op back as ``(record class name, {field: value})``, which
# ``xrexpr.intern.crossing.from_rust`` rebuilds. See ``rust/src/interned_ir.rs``.
_OpView = tuple[str, dict[str, Any]]

def roundtrip_fluent_plan(ops: list[Any]) -> list[_OpView]: ...
def roundtrip_lowered_plan(ops: list[Any]) -> list[_OpView]: ...
