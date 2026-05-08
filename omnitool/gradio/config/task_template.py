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
      - description: "Procedure description"
        outputs:                      # procedure-level schema — all dataframe columns
          - key: account_id
            kind: row                 # multi-value at read time (column extraction)
            explode: true             # one row per value at write time
            aggregate: dedup          # optional aggregate applied at finish for every execution that captures this key
            description: "account number for this user"
          - key: balance
            kind: scalar
            description: "balance shown on the account detail page"
        executions:
          - id: 1
            type: cua
            system: EPA
            inputs: [user_id]         # template scalar key references
            outputs: [account_id]     # key references into procedure outputs
            steps: |
              1. Open <user_id>'s account list.
              2. For each visible row, read_field("account_id").
          - id: 2
            type: cua
            system: EPA
            inputs: [account_id]      # runner iterates the dataframe rows
            outputs: [balance]
            steps: |
              1. Open profile for <account_id>.
              2. Read balance — capture {balance}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
    aggregate: Optional[AggregateOperation] = None

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
        return cls(
            key=data["key"],
            description=data.get("description", ""),
            format=data.get("format", ""),
            clipboard_correction=bool(data.get("clipboard_correction", True)),
            kind=kind,
            explode=bool(data.get("explode", False)),
            aggregate=aggregate,
        )


@dataclass
class TaskExecution:
    """A single execution block within a procedure."""

    type: str  # "cua" or "api"
    id: int = 0
    system: str = ""
    steps: str = ""
    inputs: List[str] = field(default_factory=list)
    resolved_outputs: List[TaskOutput] = field(default_factory=list)

    @property
    def outputs(self) -> List[str]:
        return [o.key for o in self.resolved_outputs]

    def get_resolved_output(self, key: str) -> Optional[TaskOutput]:
        """Return the TaskOutput matching *key*, or None."""
        for out in self.resolved_outputs:
            if out.key == key:
                return out
        return None

    @classmethod
    def from_dict(cls, data: dict, procedure_outputs: Iterable[TaskOutput]) -> "TaskExecution":
        raw_outputs = data.get("outputs", [])
        keys: List[str] = []
        for o in raw_outputs:
            if isinstance(o, str):
                keys.append(o)
            elif isinstance(o, dict):
                if "aggregate" in o:
                    raise ValueError(
                        f"execution {data.get('id')}: 'aggregate' must be declared on the "
                        f"procedure-level output, not in the execution outputs list."
                    )
                keys.append(o["key"])
        schema_by_key = {o.key: o for o in procedure_outputs}
        unknown = [k for k in keys if k not in schema_by_key]
        if unknown:
            raise ValueError(
                f"execution {data.get('id')}: outputs {unknown!r} not declared in "
                f"procedure outputs (have {list(schema_by_key)})."
            )
        return cls(
            id=int(data.get("id", 0)),
            type=data.get("type", "cua"),
            system=data.get("system", ""),
            steps=data.get("steps", ""),
            inputs=list(data.get("inputs", [])),
            resolved_outputs=[schema_by_key[k] for k in keys],
        )


@dataclass
class TaskProcedure:
    """A single procedure parsed from a YAML task template."""

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
        outputs = [TaskOutput.from_dict(o) for o in data.get("outputs", [])]
        executions = [TaskExecution.from_dict(e, outputs) for e in data.get("executions", [])]

        # An agent run produces one fact list per output key; two explode
        # outputs in the same execution would require an ambiguous combination
        # rule (zip vs. cartesian product). Forbid at parse time.
        explode_keys = {o.key for o in outputs if o.explode}
        for execution in executions:
            colliding = [key for key in execution.outputs if key in explode_keys]
            if len(colliding) > 1:
                raise ValueError(
                    f"Execution {execution.id}: "
                    f"{len(colliding)} explode outputs declared ({colliding}). "
                    f"At most one explode output per execution."
                )

        return cls(
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
        for key in execution.outputs:
            schema_out = procedure.get_output(key)
            desc = schema_out.description if schema_out else ""
            agg_note = (
                f" [{schema_out.aggregate.value} applied at finish]"
                if schema_out and schema_out.aggregate
                else ""
            )
            lines.append(f"  - capture {key}{agg_note}: {desc}")

    return "\n".join(lines)


@dataclass
class TaskTemplate:
    """A fully parsed task template with shared inputs and a single procedure."""

    inputs: List[TaskInput]
    procedure: TaskProcedure


def load_task_template(path: str) -> TaskTemplate:
    """Load a YAML task template and return a :class:`TaskTemplate`.

    The file must be a mapping with ``inputs`` and ``procedures`` keys::

        inputs:
          - key: input1
            value: ...
            description: ...
        procedures:
          - description: ...
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
        procedure=TaskProcedure.from_dict(raw_procs[0]),
    )


def scan_templates(directory: str | Path) -> list[tuple[str, str]]:
    """Return (label, filepath) Gradio choices for all valid templates in directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning("Templates directory not found: %s", dir_path)
        return []
    paths = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in (".yaml", ".yml"))
    choices: list[tuple[str, str]] = []
    for path in paths:
        try:
            template = load_task_template(str(path))
            label = template.procedure.description or path.stem
            choices.append((label, str(path)))
        except (yaml.YAMLError, ValueError, OSError) as exc:
            logger.warning("Skipping invalid template %s: %s", path, exc)
    return choices
