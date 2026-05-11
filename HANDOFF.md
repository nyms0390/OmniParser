# Handoff Document — OmniParser / omnitool

## Goal

Clean up the procedure/template runner system on the `refactor/enterprise-architecture` branch.
All four originally identified issues are now complete.

---

## Current Progress

### All issues done

**Issue 1 — Replace yaml_upload + procedure_dropdown with templates/ auto-scan** (`config/task_template.py`, `ui/app.py`, `ui/callbacks.py`):

- `TaskTemplate.procedures: List[TaskProcedure]` → `TaskTemplate.procedure: TaskProcedure` (singular field). The YAML format still uses a `procedures:` list key, but `load_task_template()` takes `raw_procs[0]`.
- `scan_templates(directory)` added to `config/task_template.py` — globs `.yaml`/`.yml`, skips invalid files, returns `[(description, filepath)]` for Gradio.
- `yaml_upload` (gr.File) and `procedure_dropdown` removed from `app.py`. Replaced with `template_dropdown` (gr.Dropdown) populated at startup via `scan_templates(_TEMPLATES_DIR)`.
- `_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "templates"` — module-level constant in `app.py`.
- `on_yaml_upload` → `on_template_select(filepath: Optional[str])` in `callbacks.py`: takes filepath string, returns 2 values `(template, exec_dropdown_update)`.
- `on_procedure_change` removed entirely.
- `on_submit`: `selected_procedure_idx` parameter removed; always uses `yaml_template.procedure`.

**Issue 2 — ProcedureRunner terminal event cleanup** (`core/procedure_runner.py`, `ui/callbacks.py`):

- `run_execution()` removed entirely.
- `run()` renamed `run_procedure()`.
- New public `run_once(execution, row_idx=0)` — pure agent pass-through, yields `complete` with facts/cost, no dataframe merge. UI "test a single execution" entry point.
- `execution_complete` in callbacks is mid-run progress marker only — `loop_complete = True` removed.

**Issue 4 — Strip `task_procedure` from `create_agent`; pass only `task_execution`** (`config/task_template.py`, `core/procedure_runner.py`, `core/agents/base.py`, `core/agents/react_agent.py`, `core/agents/factory.py`, `ui/callbacks.py`, tests):

- Added `ResolvedOutput` dataclass to `task_template.py`: merges `TaskOutput` fields (`kind`, `clipboard_correction`, `description`) with `ExecutionOutput.aggregate`. Added `resolved_outputs: List[ResolvedOutput]` field and `get_resolved_output()` method to `TaskExecution`.
- `ProcedureRunner._resolve_execution()` pre-resolves outputs from the procedure schema — called once per execution in `run_procedure()` (not per row), and once in `run_once()`.
- `run_once()` guards against out-of-range `row_idx` and yields a structured `error` event instead of letting `IndexError` propagate through the Gradio stream.
- `_run_once()` expects a pre-resolved execution from the caller; no redundant work per row.
- `BaseAgent.__init__()`: removed `task_procedure` parameter. `_handle_read_field` and `_handle_save_field` now read from `task_execution.get_resolved_output()` — single lookup replaces two separate lookups.
- `create_agent()` and `factory.py`: `task_procedure` parameter removed.
- `callbacks.py`: `task_procedure` removed from `orchestrator_kwargs`; `base_kwargs` indirection eliminated; `agent_factory` closure simplified.
- Tests updated: `_helpers.py`, `test_handle_read_field.py`, `test_screenshot_saving.py`.

**All 311/311 tests pass.**

---

## What Worked

- Collapsing `procedures: List` → `procedure: TaskProcedure` directly (no transition layer) — all 311 tests updated mechanically, no ambiguity left.
- `scan_templates` using `iterdir()` with suffix filter for a unified sorted pass.
- Guard for missing `templates/` directory returns `[]` instead of raising.
- Empty `procedure.description` falls back to `path.stem` as dropdown label.
- `ResolvedOutput` as a flat merged struct: one lookup in the agent replaces two separate lookups against `task_procedure` and `task_execution`.
- Hoisting `_resolve_execution` to once-per-execution (not once-per-row) avoids redundant O(M) schema scans in the hot path.

## What Didn't Work

- Nothing failed.

---

## Key Files

| File | Purpose |
|------|---------|
| `omnitool/gradio/config/task_template.py` | `TaskTemplate.procedure`, `ResolvedOutput`, `TaskExecution.resolved_outputs`, `scan_templates` |
| `omnitool/gradio/ui/app.py` | `template_dropdown`, `_TEMPLATES_DIR`, event wiring |
| `omnitool/gradio/ui/callbacks.py` | `on_template_select`, `on_mode_change`, `on_submit`, `agent_factory` closure |
| `omnitool/gradio/core/procedure_runner.py` | `run_procedure`, `run_once`, `_run_once`, `_resolve_execution`, `_merge_facts` |
| `omnitool/gradio/core/agents/factory.py` | `create_agent` — `task_procedure` removed |
| `omnitool/gradio/core/agents/base.py` | `_handle_read_field`, `_handle_save_field` — reads `task_execution.get_resolved_output()` |
| `templates/` | Template YAML files (repo root); currently `test_tamplate.yaml` |
| `omnitool/gradio/tests/test_procedure_runner.py` | 25 tests |
| `omnitool/gradio/tests/test_ui_callbacks.py` | 72 tests |
| `omnitool/gradio/tests/test_handle_read_field.py` | Tests for kind-aware read_field behavior |
| `omnitool/gradio/tests/test_screenshot_saving.py` | Tests for save_field / aggregate behavior |

## Run Tests

```bash
conda run -n omni pytest omnitool/gradio/tests/ -q --ignore=omnitool/gradio/tests/test_core_components.py
```

(`test_core_components.py` has a pre-existing `ImportError` on `get_model_config` — always exclude.)

---

## Prior Session Context (VLMAgent refactor — completed)

- `ReActAgent` is the active agent; `VLMAgent`/`OmniAgent`/`GTAAgent` legacy names route to it via factory
- `GroundingStrategy` is pluggable: `OmniParserGrounding` (SOM box_id) or `GTA1Grounding` (natural-language target)
- `CLAUDE.md` has architecture overview and dev commands

---

## Open Code Review Findings (not yet addressed)

From the `/improve` review of the Issue 4 changes — three items were noted but not acted on:

1. **`_apply_template_aggregates` reads `task_execution.outputs` not `resolved_outputs`** — functionally correct (`aggregate` lives in both places), but inconsistent with the new pattern. Low risk; document or migrate when convenient.
2. **Silent fallback in `callbacks.py`** — when `execution_selection` doesn't match any execution id, `run_procedure()` (full multi-agent run) fires silently instead of surfacing an error. Pre-existing behavior; low priority.
3. **`ResolvedOutput.merge(exec_out, schema_out)` classmethod** — suggested to co-locate merge logic on the dataclass. Purely ergonomic.
