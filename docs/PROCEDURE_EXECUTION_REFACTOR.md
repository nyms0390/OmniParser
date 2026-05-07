# Procedure Execution Refactor

## Goal

Change template execution scale from per-procedure (today) to per-execution, with a
row-oriented procedure-scoped dataframe that:

1. Carries scalar parameters across executions inside one procedure run (e.g. `user_id` from template inputs).
2. Builds up a table of rows, where one execution can create rows (e.g. one row per account discovered) and a later execution iterates those rows to add columns (e.g. balance, status per account).
3. Lets executions in the same procedure target different systems (today's `system_config` is bound once from `executions[0].system`).

A new `ProcedureRunner` drives multiple `TaskExecution`s sequentially. The agent stays dumb — one `TaskExecution` (or one iteration of a row-iterating execution) = one agent run.

---

## Data Model

### Procedure-level schema (`TaskOutput`)

`TaskProcedure.outputs` declares **all dataframe columns** — their key, kind, description, and format. No `aggregate` here; the procedure sees only the final reduced value per cell.

Fields:
- `key: str`
- `kind: ColumnKind` — `scalar` (one value per row) or `row` (values become row identifiers)
- `description: str`
- `format: str`
- `clipboard_correction: bool`

### Execution-level output refs (`ExecutionOutput`)

`TaskExecution.outputs` is a list of `ExecutionOutput` — key references into the procedure schema, plus an optional intra-execution aggregate.

Fields:
- `key: str` — must match a `TaskOutput.key` on the parent procedure
- `aggregate: Optional[AggregateOperation]` — reduction applied to multiple `read_field` calls **within one agent run** before writing to the dataframe

### Execution-level inputs (`TaskExecution.inputs`)

`TaskExecution.inputs: List[str]` — key references. `ProcedureRunner` resolves each key:
- Found in `TaskTemplate.inputs` → scalar, substitute value into steps; execution runs once.
- Found in procedure schema with `kind: row` → row column; execution iterates once per dataframe row, substituting that row's value into steps.

### Aggregate semantics

Aggregate is an **intra-execution** reduction: multiple `read_field` calls within one agent run → reduced → one value written to the dataframe. It applies to both kinds:

| kind   | aggregate  | meaning |
|--------|------------|---------|
| scalar | —          | one read → one column value |
| scalar | sum/concat | N reads → reduced → one column value |
| row    | —          | N reads → N new rows |
| row    | dedup      | N reads → deduplicated → M new rows |

### `TaskProcedure` changes

- **Remove**: `inputs`, `_substitute_inputs`, `_cua_steps`, `_referenced_outputs`, `to_task_string`, `to_extract_fields`
- **Keep**: `id`, `description`, `outputs` (schema), `executions`, `get_output(key) -> Optional[TaskOutput]`

### `TaskTemplate` changes

- **Remove**: `resolve_inputs(procedure)`
- **Add**: `resolve_execution_inputs(execution) -> Dict[str, str]` — returns `{key: value}` for execution input keys found in `template.inputs`

---

## Worked example

```yaml
inputs:
  - key: user_id
    value: 12345

procedures:
  - ID: 1
    description: Pull all accounts for a user and enrich each with balance + status.

    outputs:                              # procedure-level schema — all dataframe columns
      - key: account_id
        kind: row
        description: account number for this user
      - key: balance
        kind: scalar
        description: balance shown on the account detail page
      - key: status
        kind: scalar
        description: account status

    executions:
      - id: 1
        type: cua
        system: EPA
        inputs: [user_id]                 # template scalar → runs once
        outputs:
          - key: account_id
            aggregate: dedup             # N reads → deduplicated → M rows
        steps: |
          1. Open user <user_id>'s account list.
          2. For each visible row, read_field("account_id").

      - id: 2
        type: cua
        system: EPA
        inputs: [account_id]             # kind: row in schema → runner iterates per row
        outputs:
          - key: balance                  # no aggregate — one read per agent run
          - key: status
        steps: |
          1. Open profile for <account_id>.
          2. Read balance — capture {balance}.
          3. Read status — capture {status}.
```

Result dataframe exported as CSV:

| account_id | balance | status |
|------------|---------|--------|
| A001       | 10.00   | active |
| A002       | 20.00   | closed |
| A003       | 30.00   | active |

---

## Implementation Plan

### Step 1 — `config/enums.py`: renames ✓

- Rename `FieldKind` → `ColumnKind` (values `SCALAR` / `ROW` unchanged).
- Update all imports and references across `task_template.py`, `base.py`, `schemas.py`, and tests.

Verify: `grep -r FieldKind` returns no hits outside of enums.py history.

---

### Step 2 — `config/task_template.py`: data model

- **Remove** `TaskOutputAggregate` entirely. Its only field was `operation: AggregateOperation`; callers now hold `AggregateOperation` directly.
- **Remove** `aggregate: Optional[TaskOutputAggregate]` from `TaskOutput` and its `from_dict`.
- **Add** `ExecutionOutput` dataclass:
  ```python
  @dataclass
  class ExecutionOutput:
      key: str
      aggregate: Optional[AggregateOperation] = None

      @classmethod
      def from_dict(cls, data) -> "ExecutionOutput":
          if isinstance(data, str):          # outputs: [balance, status]
              return cls(key=data)
          return cls(
              key=data["key"],
              aggregate=AggregateOperation(data["aggregate"]) if data.get("aggregate") else None,
          )
  ```
- **Expand** `TaskExecution`: add `id: int`, `inputs: List[str]`, `outputs: List[ExecutionOutput]`; add `get_output(key) -> Optional[ExecutionOutput]`; update `from_dict`.
- **Remove** from `TaskProcedure`: `inputs`, `_substitute_inputs`, `_cua_steps`, `_referenced_outputs`, `to_task_string`, `to_extract_fields`.
- **Add** module-level `build_execution_task_string(execution: TaskExecution, procedure: TaskProcedure, resolved_inputs: Dict[str, str]) -> str`.
- **Remove** `TaskTemplate.resolve_inputs()`; **add** `TaskTemplate.resolve_execution_inputs(execution: TaskExecution) -> Dict[str, str]`.

Verify: `load_task_template` parses the new schema; `build_execution_task_string` produces correct per-execution task string.

---

### Step 3 — `core/agents/base.py`: split aggregate and schema lookups

- Add `task_execution: Optional[TaskExecution] = None` to `__init__`.
- Fix `system_config` resolution: `get_system_config(task_execution.system)` when `task_execution` is set, otherwise `None`.
- `_handle_read_field`: `kind`/`clipboard_correction` from `task_procedure.get_output(field_name)`; `aggregate` from `task_execution.get_output(field_name)`.
- `_handle_save_field`: `accumulates = (exec_out.aggregate is not None) or (schema_out.kind == ColumnKind.ROW)` where `exec_out` from `task_execution` and `schema_out` from `task_procedure`.
- `_apply_template_aggregates`: iterate `task_execution.outputs` (not `task_procedure.outputs`); when `task_execution` is `None`, return early.

Interactive mode (both `None`): behaviour unchanged — `kind` defaults to `SCALAR`, `clipboard_correction` to `True`, `aggregate` to `None`.

Verify: `test_handle_read_field.py` and `test_base_agent.py` pass with updated fixtures.

---

### Step 4 — `core/agents/factory.py`: thread `task_execution`

- Add `task_execution: Optional[TaskExecution] = None` parameter.
- Pass through to `ReActAgent` (which forwards to `BaseAgent`).

Verify: `test_factory.py` passes.

---

### Step 5 — `core/procedure_runner.py`: new orchestrator

New file. Two classes:

**`ProcedureDataframe`**:
- `rows: List[Dict[str, str]]`
- `add_rows(key: str, values: List[str])` — appends `{key: v}` for each value (row outputs)
- `set_cell(row_idx: int, key: str, value: str)` — writes scalar into existing row
- `to_csv() -> str`

**`ProcedureRunner(procedure, template, agent_factory, save_folder)`**:

`run()` generator:
1. For each `execution` in `procedure.executions`:
   - `row_input_key` = first key in `execution.inputs` whose procedure schema `kind == ColumnKind.ROW`, or `None`.
   - `scalar_inputs = template.resolve_execution_inputs(execution)`.
   - If `row_input_key is None`: one agent run; pass `scalar_inputs` to `build_execution_task_string`.
   - Else: for each `(row_idx, row)` in `enumerate(dataframe.rows)`: one agent run with `row[row_input_key]` + `scalar_inputs` substituted.
   - Each sub-run: forward all agent events except `complete`; on `complete`, write `facts` into dataframe (`add_rows` for `ColumnKind.ROW` keys, `set_cell` for `ColumnKind.SCALAR` keys).
   - Yield `{"type": "execution_complete", "execution_id": execution.id}`.
2. Write `procedure_{id}_result.csv` to `save_folder`.
3. Yield `{"type": "procedure_complete", "csv_path": str, "rows": dataframe.rows}`.

Verify: unit test with the 2-execution example template; dataframe matches expected table.

---

### Step 6 — `ui/callbacks.py`: route TASK mode to `ProcedureRunner`

- TASK mode: instantiate `ProcedureRunner`; drive `runner.run()` generator instead of a single agent.
- Remove `message = yaml_procedure.to_task_string()` in TASK mode (runner handles task strings per execution).
- Handle `execution_complete`: `yield history, "", f"Execution {id} complete", state`.
- Handle `procedure_complete`: render `rows` as an HTML table; append CSV path link; yield final state.
- Keep the existing `complete` handler for interactive mode.

Verify: `test_ui_callbacks.py` passes.

---

### Step 7 — Tests

- `test_core_components.py`: remove procedure `inputs` from fixtures; add `ExecutionOutput` in execution fixtures; test `build_execution_task_string`.
- `test_handle_read_field.py`: add `task_execution` with `ExecutionOutput`s to agent fixture; verify aggregate comes from execution, kind from procedure schema.
- `test_base_agent.py`: pass `task_execution` to constructor fixtures.
- New `tests/test_procedure_runner.py`: test `ProcedureDataframe.add_rows`, `set_cell`, `to_csv`; test `ProcedureRunner.run()` with a mock agent factory producing synthetic `complete` events.

Verify: `conda run -n omni pytest omnitool/gradio/tests/` all green.
