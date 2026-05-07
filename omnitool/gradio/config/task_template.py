"""YAML task template loader for TASK mode.

Parses a YAML file with top-level ``inputs`` and ``procedures`` keys.

YAML schema::

    inputs:
      - key: user_id
        value: 12345
        description: "..."
        format: 8 number digits
        required: true

    procedures:
      - ID: 1
        description: "Procedure description"
        outputs:                      # procedure-level schema — all dataframe columns
          - key: account_id
            kind: row                 # multi-value at read time (column extraction)
            explode: true             # one row per value at write time
            description: "account number for this user"
          - key: balance
            kind: scalar
            description: "balance shown on the account detail page"
        executions:
          - id: 1
            type: cua
            system: EPA
            inputs: [user_id]         # template scalar key references
            outputs:
              - key: account_id
                aggregate: dedup      # optional intra-execution aggregate
            steps: |
              1. Open <user_id>'s account list.
              2. For each visible row, read_field("account_id").
          - id: 2
            type: cua
            system: EPA
            inputs: [account_id]      # runner iterates the dataframe rows
            outputs:
              - key: balance
            steps: |
              1. Open profile for <account_id>.
              2. Read balance — capture {balance}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from omnitool.gradio.config.enums import AggregateOperation, ColumnKind

logger = logging.getLogger(__name__)


@dataclass
class TaskInput:
    """A single input parameter defined at the template level."""

    key: str
    description: str = ""
    format: str = ""
    required: bool = False
    value: Optional[str] = None

    @classmethod
    def from_dict(cls, data) -> "TaskInput":
        if isinstance(data, str):
            return cls(key=data)
        return cls(
            key=data["key"],
            description=data.get("description", ""),
            format=data.get("format", ""),
            required=bool(data.get("required", False)),
            value=str(data["value"]) if data.get("value") is not None else None,
        )


@dataclass
class TaskOutput:
    """A single output field declared in a procedure's schema.

    ``kind`` controls read-time semantics (SCALAR → clipboard correction,
    ROW → column extraction). ``explode`` controls write-time semantics
    (True → one dataframe row per value; False → values joined into one
    cell). The two flags are orthogonal.
    """

    key: str
    description: str = ""
    format: str = ""
    clipboard_correction: bool = True
    kind: ColumnKind = ColumnKind.SCALAR
    explode: bool = False

    @classmethod
    def from_dict(cls, data) -> "TaskOutput":
        if isinstance(data, str):
            return cls(key=data)
        raw_kind = data.get("kind", ColumnKind.SCALAR.value)
        try:
            kind = ColumnKind(raw_kind)
        except ValueError:
            valid = [k.value for k in ColumnKind]
            raise ValueError(
                f"Invalid kind {raw_kind!r} for output {data.get('key')!r}. "
                f"Valid values: {valid}"
            ) from None
        return cls(
            key=data["key"],
            description=data.get("description", ""),
            format=data.get("format", ""),
            clipboard_correction=bool(data.get("clipboard_correction", True)),
            kind=kind,
            explode=bool(data.get("explode", False)),
        )


@dataclass
class ExecutionOutput:
    """An output key reference within a TaskExecution, with an optional intra-execution aggregate."""

    key: str
    aggregate: Optional[AggregateOperation] = None

    @classmethod
    def from_dict(cls, data) -> "ExecutionOutput":
        if isinstance(data, str):
            return cls(key=data)
        raw_agg = data.get("aggregate")
        if raw_agg is not None:
            try:
                aggregate = AggregateOperation(raw_agg)
            except ValueError:
                valid = [o.value for o in AggregateOperation]
                raise ValueError(
                    f"Invalid aggregate operation {raw_agg!r}. Valid values: {valid}"
                ) from None
        else:
            aggregate = None
        return cls(key=data["key"], aggregate=aggregate)


@dataclass
class TaskExecution:
    """A single execution block within a procedure."""

    type: str  # "cua" or "api"
    id: int = 0
    system: str = ""
    steps: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[ExecutionOutput] = field(default_factory=list)

    def get_output(self, key: str) -> Optional[ExecutionOutput]:
        """Return the ExecutionOutput matching *key*, or None."""
        for out in self.outputs:
            if out.key == key:
                return out
        return None

    @classmethod
    def from_dict(cls, data: dict) -> "TaskExecution":
        return cls(
            id=int(data.get("id", 0)),
            type=data.get("type", "cua"),
            system=data.get("system", ""),
            steps=data.get("steps", ""),
            inputs=list(data.get("inputs", [])),
            outputs=[ExecutionOutput.from_dict(o) for o in data.get("outputs", [])],
        )


@dataclass
class TaskProcedure:
    """A single procedure parsed from a YAML task template."""

    id: int
    description: str
    outputs: List[TaskOutput] = field(default_factory=list)
    executions: List[TaskExecution] = field(default_factory=list)

    def get_output(self, field_name: str) -> Optional[TaskOutput]:
        """Return the TaskOutput matching *field_name*, or None."""
        for out in self.outputs:
            if out.key == field_name:
                return out
        return None

    @classmethod
    def from_dict(cls, data: dict) -> "TaskProcedure":
        if "ID" not in data and "id" not in data:
            raise ValueError("Procedure dict missing 'ID' field.")
        proc_id = data.get("ID", data.get("id"))
        outputs = [TaskOutput.from_dict(o) for o in data.get("outputs", [])]
        executions = [TaskExecution.from_dict(e) for e in data.get("executions", [])]

        # An agent run produces one fact list per output key; two explode
        # outputs in the same execution would require an ambiguous combination
        # rule (zip vs. cartesian product). Forbid at parse time.
        explode_keys = {o.key for o in outputs if o.explode}
        for execution in executions:
            colliding = [eo.key for eo in execution.outputs if eo.key in explode_keys]
            if len(colliding) > 1:
                raise ValueError(
                    f"Procedure {proc_id} execution {execution.id}: "
                    f"{len(colliding)} explode outputs declared ({colliding}). "
                    f"At most one explode output per execution."
                )

        return cls(
            id=int(proc_id),
            description=data.get("description", ""),
            outputs=outputs,
            executions=executions,
        )


def build_execution_task_string(
    execution: TaskExecution,
    procedure: TaskProcedure,
    resolved_inputs: Dict[str, str],
) -> str:
    """Build a per-execution task string for the agent.

    Substitutes *resolved_inputs* values into ``<key>`` step placeholders and
    lists declared outputs with their aggregate annotations.
    """
    lines = [procedure.description]

    if resolved_inputs:
        lines.append("\nInputs:")
        for key, value in resolved_inputs.items():
            lines.append(f"  - {key}: {value}")

    if execution.steps:
        steps = execution.steps.strip()
        for key, value in resolved_inputs.items():
            steps = steps.replace(f"<{key}>", value)
        lines.append("\nSteps:")
        lines.append(steps)

    if execution.outputs:
        lines.append("\nOutputs:")
        for exec_out in execution.outputs:
            schema_out = procedure.get_output(exec_out.key)
            desc = schema_out.description if schema_out else ""
            agg_note = (
                f" [{exec_out.aggregate.value} applied at finish]"
                if exec_out.aggregate
                else ""
            )
            lines.append(f"  - capture {exec_out.key}{agg_note}: {desc}")

    return "\n".join(lines)


@dataclass
class TaskTemplate:
    """A fully parsed task template with shared inputs and procedures."""

    inputs: List[TaskInput]
    procedures: List[TaskProcedure]

    def get_procedure(self, proc_id: int) -> TaskProcedure:
        """Return the procedure with the given ID, or raise :class:`ValueError`."""
        for p in self.procedures:
            if p.id == proc_id:
                return p
        raise ValueError(f"Procedure ID {proc_id} not found in template.")


def load_task_template(path: str) -> TaskTemplate:
    """Load a YAML task template and return a :class:`TaskTemplate`.

    The file must be a mapping with ``inputs`` and ``procedures`` keys::

        inputs:
          - key: input1
            value: ...
            description: ...
        procedures:
          - ID: 1
            description: ...
            ...

    Args:
        path: Filesystem path to the ``.yaml`` / ``.yml`` file.

    Returns:
        :class:`TaskTemplate` with all inputs and procedures parsed.

    Raises:
        ValueError: If the file is empty, not in the expected format, or has
            no procedures.
        yaml.YAMLError: If the file cannot be parsed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "procedures" not in data:
        raise ValueError(
            f"Task template '{path}' must be a mapping with 'inputs' and 'procedures' keys."
        )

    raw_inputs = data.get("inputs", [])
    raw_procs = data["procedures"]

    if not isinstance(raw_procs, list) or not raw_procs:
        raise ValueError(
            f"Task template '{path}': 'procedures' must be a non-empty list."
        )

    return TaskTemplate(
        inputs=[TaskInput.from_dict(i) for i in raw_inputs],
        procedures=[TaskProcedure.from_dict(p) for p in raw_procs],
    )
