# Planning and design notes

Internal working documents: how the design got where it is, and what is planned next.
**None of this is user documentation** — that lives in `docs/`, which is the Sphinx site
and nothing else.

Expect these to be uneven. Some are settled design rationale, some are transcripts of
arguments, and some are superseded by the code they were arguing about. Where a memo and
the code disagree, the code wins.

## The roadmap

[`roadmap/`](roadmap/) is the live plan, one workstream per file. Start with
[`roadmap/00-assessment.md`](roadmap/00-assessment.md): it states where the codebase
stands, what is missing, and the recommended order for what remains.

## The design memos

| file | what it argues |
|---|---|
| [`structural-dispatch.md`](structural-dispatch.md) | why operations dispatch on a `kind` sum type rather than on method names |
| [`structural-dispatch-2.md`](structural-dispatch-2.md) | the follow-up, after the first pass met real code |
| [`structural-dispatch-discussion.md`](structural-dispatch-discussion.md) | the discussion behind both |
| [`xrexpr_ir_discussion.md`](xrexpr_ir_discussion.md) | how the IR came to be a flat list of fat variants |
| [`xrexpr_ir_discussion_expanded.md`](xrexpr_ir_discussion_expanded.md) | the expanded version of the same |
| [`xrexpr_roadmap_review_discussion.md`](xrexpr_roadmap_review_discussion.md) | a review pass over the roadmap itself |
| [`improvement-report.md`](improvement-report.md) | a survey of what wanted fixing |
| [`pr-plan.md`](pr-plan.md) | how that survey was cut into landable PRs |
| [`indexer-follow-ups.md`](indexer-follow-ups.md) | loose ends left by the indexer taxonomy |

## Citing these from code

Source comments cite these files by name — `` ``07-small-wins.md`` §8 `` — and a few cite
the path. Both forms mean a file under this directory. If you move one, grep for its
bare filename as well as its path.
