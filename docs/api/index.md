# API reference

Everything on this page is generated from the source: the signatures are the real ones
and the prose is the docstring, so nothing here can drift from the code. The short notes
under each heading are orientation only — the pages that *explain* the design are under
[Internals](../internals/pipeline.md).

The grouping follows the modules, and within a module it follows `__all__`. Names not
listed here are private, and the fact that a helper appears in an internals page's prose
does not make it public.

## The public surface

Three names, and only the first is exported from the package root. `LazyProxy` is what
`ds.plan` returns, so its methods are the ones you actually call; you never construct one
yourself.

```{eval-rst}
.. currentmodule:: xrexpr

.. autosummary::
   :toctree: generated/
   :nosignatures:

   InvalidExpressionError
```

```{eval-rst}
.. currentmodule:: xrexpr.accessor

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LazyProxy
   Explanation
```

## The plan

The intermediate representation: a flat list of frozen dataclasses. The six shared
variants appear at both levels, `ContextOpen` only before lowering, and the three fused
kinds only after it — see [the IR](../internals/ir.md) for why the levels differ and why the
list is a list.

```{eval-rst}
.. currentmodule:: xrexpr.ir

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Reduce
   Select
   Scan
   Project
   Rechunk
   Opaque
   ContextOpen
   GroupedReduce
   WindowedReduce
   WeightedReduce
```

The unions and the vocabulary they are built from. `ALL_DIMS` is a sentinel meaning *every
dim*, deliberately distinct from `None`, which means *unknown*.

```{eval-rst}
.. currentmodule:: xrexpr.ir

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Op
   FluentOp
   LoweredOp
   DimSet
   AllDims
   ALL_DIMS
   ContextOpenName
   frozendict
```

## Recognising a call

The table that turns a method name into a node kind. Each spec carries the dispatch
behaviour for its kind, so adding a method is a table entry rather than a branch — see
[the operation table](../internals/operations.md).

```{eval-rst}
.. currentmodule:: xrexpr.operations

.. autosummary::
   :toctree: generated/
   :nosignatures:

   OpSpec
   ReduceSpec
   ScanSpec
   SelectSpec
   RechunkSpec
   ProjectSpec
   ContextSpec
   OP_TABLE
   CONTEXT_METHODS
   spec
```

## The logical schema

What the optimiser knows about the data without looking at it: dim sizes and variable
names, folded forward one node at a time. See [the logical schema](../internals/schema.md).

```{eval-rst}
.. currentmodule:: xrexpr.schema

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SchemaState
   to_opnode
   apply_schema
   resolve_dims
```

## Lowering and emission

The two translation boundaries either side of the optimiser. See
[lowering](../internals/lowering.md).

```{eval-rst}
.. currentmodule:: xrexpr.lower

.. autosummary::
   :toctree: generated/
   :nosignatures:

   to_lower_ir
   emit
   Call
```

## The optimiser

`optimize` is the entry point; the five rules below are what it runs to a fixpoint, and
`dim_effect` is the single site both pushdown rules consult. See
[the optimiser](../internals/optimiser.md).

```{eval-rst}
.. currentmodule:: xrexpr.optimize

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optimize
   Plan
   merge_adjacent_selects
   merge_adjacent_projects
   pushdown_selects
   pushdown_projections
   pushdown_selects_past_rechunks
   DimEffect
   dim_effect
```

## Indexers

What a `sel` or `isel` argument turns out to be, and what that implies for the dimension
it names. `classify` is the sole constructor and is total. See
[the value taxonomies](../internals/values.md).

```{eval-rst}
.. currentmodule:: xrexpr.indexers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Indexer
   Scalar
   ForwardSlice
   GeneralSlice
   Positions
   Mask
   Label
   classify
```

## Chunk specs

The same shape, for `chunk()` arguments. `classify_chunk` is the sole constructor, and
the distinction the optimiser matches on is whether a spec's meaning depends on the
extent of the dimension it is applied to.

```{eval-rst}
.. currentmodule:: xrexpr.chunks

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ChunkSpec
   SingleSize
   Auto
   ByteSize
   FullDim
   NoChange
   BlockSeq
   OpaqueChunk
   classify_chunk
```

## Rendering a plan

What `.explain()` calls. See [reading explain output](../guide/reading-explain.md).

```{eval-rst}
.. currentmodule:: xrexpr.explain

.. autosummary::
   :toctree: generated/
   :nosignatures:

   format_plan
```
