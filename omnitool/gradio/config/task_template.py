"""YAML task template loader for TASK mode.

Parses a YAML file with top-level ``inputs`` and ``procedures`` keys.

YAML schema::

    inputs:
      - key: input1
        value: 12345678
        description: "..."
        format: 8 number digits
        required: true

    procedures:
      - ID: 1
        description: "Procedure description"
        inputs: [input1, input2]    # key references only
        outputs:
          - key: output1
            description: "..."
            format: string
        executions:
          - type: cua
            system: EPA
            steps: |
              1. Step one using <input1>.
              2. Step two — record the result as {output1}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional

import yaml

from omnitool.gradio.config.enums import AggregateOperation

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
class TaskOutputAggregate:
    """Aggregate computation declared on a TaskOutput.

    When present, the agent computes this aggregate at finish time by reading
    the accumulated values of *source* from working_memory.facts and writing
    the result under the parent output's key.
    """

    operation: AggregateOperation
    source: str     # key of the dynamic field whose values are aggregated

    @classmethod
    def from_dict(cls, data: dict) -> "TaskOutputAggregate":
        try:
            op = AggregateOperation(data["operation"])
        except ValueError:
            valid = [o.value for o in AggregateOperation]
            raise ValueError(
                f"Invalid aggregate operation {data['operation']!r}. Valid values: {valid}"
            ) from None
        return cls(operation=op, source=data["source"])


@dataclass
class TaskOutput:
    """A single output field produced by a procedure."""

    key: str
    description: str = ""
    format: str = ""
    clipboard_correction: bool = True
    dynamic: bool = False
    aggregate: Optional[TaskOutputAggregate] = None

    @property
    def is_dynamic(self) -> bool:
        """True when values accumulate rather than overwrite.

        A field is dynamic when explicitly flagged, or when it is the source
        of an aggregate output (those always accumulate).
        """
        return self.dynamic or self.aggregate is not None

    @classmethod
    def from_dict(cls, data) -> "TaskOutput":
        if isinstance(data, str):
            return cls(key=data)
        return cls(
            key=data["key"],
            description=data.get("description", ""),
            format=data.get("format", ""),
            clipboard_correction=bool(data.get("clipboard_correction", True)),
            dynamic=bool(data.get("dynamic", False)),
            aggregate=(
                TaskOutputAggregate.from_dict(raw_agg)
                if (raw_agg := data.get("aggregate"))
                else None
            ),
        )


@dataclass
class TaskExecution:
    """A single execution block within a procedure."""

    type: str  # "cua" or "api"
    system: str = ""
    steps: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "TaskExecution":
        return cls(
            type=data.get("type", "cua"),
            system=data.get("system", ""),
            steps=data.get("steps", ""),
        )


@dataclass
class TaskProcedure:
    """A single procedure parsed from a YAML task template."""

    id: int
    description: str
    inputs: List[TaskInput] = field(default_factory=list)
    outputs: List[TaskOutput] = field(default_factory=list)
    executions: List[TaskExecution] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _substitute_inputs(self, text: str) -> str:
        """Replace ``<key>`` placeholders in *text* with resolved input values."""
        for inp in self.inputs:
            if inp.value is not None:
                text = text.replace(f"<{inp.key}>", str(inp.value))
        return text

    def _cua_steps(self) -> str:
        """Return the combined steps text from all CUA executions."""
        parts = []
        for execution in self.executions:
            if execution.type.lower() == "cua" and execution.steps:
                parts.append(execution.steps.strip())
        return "\n".join(parts)

    def _referenced_outputs(self, steps_text: str) -> List[TaskOutput]:
        """Return outputs whose ``{key}`` placeholder appears in *steps_text*."""
        return [out for out in self.outputs if f"{{{out.key}}}" in steps_text]

    def to_task_string(self) -> str:
        """Build a task string suitable for ``_init_checklist_from_procedure``.

        The string contains:
        - Procedure description
        - Inputs summary (key: value or N/A) for resolved inputs
        - Numbered steps from CUA executions (with input values substituted)
        - Outputs to capture block for any ``{key}`` output references in steps

        The numbered steps are embedded so that
        ``Checklist.from_user_text()`` can parse them into checklist items.
        """
        lines = [self.description]

        if self.inputs:
            lines.append("\nInputs:")
            for inp in self.inputs:
                val = str(inp.value) if inp.value is not None else "N/A"
                lines.append(f"  - {inp.key}: {val}")

        steps_raw = self._cua_steps()
        if steps_raw:
            lines.append("\nSteps:")
            substituted = self._substitute_inputs(steps_raw)
            for raw_line, display_line in zip(steps_raw.splitlines(), substituted.splitlines()):
                lines.append(display_line)
                # Match against raw_line: {key} placeholders intact before substitution
                for out in self._referenced_outputs(raw_line):
                    if out.aggregate:
                        lines.append(
                            f"   capture: {out.aggregate.source}"
                            f"  [{out.key} is auto-computed as"
                            f" {out.aggregate.operation.value} at finish — do not capture directly]"
                        )
                    else:
                        lines.append(f"   capture: {out.key}")

        capturable = [o for o in self.outputs if o.aggregate is None]
        auto_computed = [o for o in self.outputs if o.aggregate is not None]
        capturable_keys = {o.key for o in capturable}
        if capturable or auto_computed:
            lines.append("\nOutputs:")
            for out in capturable:
                mode = " (one entry per row)" if out.dynamic else ""
                lines.append(f"  - capture {out.key}{mode}: {out.description}")
            for out in auto_computed:
                if out.aggregate.source not in capturable_keys:
                    lines.append(
                        f"  - capture {out.aggregate.source} (one entry per row):"
                        f" {out.description}"
                        f"  [{out.key} will be auto-computed as"
                        f" {out.aggregate.operation.value} of {out.aggregate.source}]"
                    )
                else:
                    lines.append(
                        f"  - {out.key} will be auto-computed as"
                        f" {out.aggregate.operation.value} of {out.aggregate.source}"
                    )

        return "\n".join(lines)

    def to_extract_fields(self) -> Dict[str, str]:
        """Return ``{output.key: output.description}`` for directly-capturable outputs only.

        Aggregate outputs are excluded — they are auto-computed at finish and
        must not be captured directly by the LLM.
        """
        return {out.key: out.description for out in self.outputs if out.aggregate is None}

    @cached_property
    def _output_index(self) -> Dict[str, "TaskOutput"]:
        """Map each field name (or aggregate source key) to its TaskOutput, built once."""
        index: Dict[str, TaskOutput] = {}
        for out in self.outputs:
            key = out.key if out.aggregate is None else out.aggregate.source
            index.setdefault(key, out)
        return index

    def get_output(self, field_name: str) -> Optional[TaskOutput]:
        """Return the TaskOutput matching *field_name* (or its aggregate source), or None.

        Callers read attributes directly (``out.is_dynamic``,
        ``out.clipboard_correction``) and apply their own defaults when the
        field is unknown.
        """
        return self._output_index.get(field_name)

    # ------------------------------------------------------------------
    # Class-level constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "TaskProcedure":
        if "ID" not in data and "id" not in data:
            raise ValueError("Procedure dict missing 'ID' field.")
        proc_id = data.get("ID", data.get("id"))
        raw_inputs = data.get("inputs", [])
        # Accept both plain strings and dicts with a "key" field; create stubs
        # (no value) that resolve_inputs() will later fill in.
        input_stubs = [
            TaskInput(key=i) if isinstance(i, str) else TaskInput(key=i["key"])
            for i in raw_inputs
        ]
        return cls(
            id=int(proc_id),
            description=data.get("description", ""),
            inputs=input_stubs,
            outputs=[TaskOutput.from_dict(o) for o in data.get("outputs", [])],
            executions=[TaskExecution.from_dict(e) for e in data.get("executions", [])],
        )


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

    def resolve_inputs(self, procedure: TaskProcedure) -> None:
        """Resolve input values into *procedure* in-place.

        Replaces each stub in ``procedure.inputs`` with the corresponding
        :class:`TaskInput` from the template-level inputs (which carry values).
        Keys not found in the top-level inputs produce a warning and are left
        as stubs.
        """
        input_map = {i.key: i for i in self.inputs}
        for idx, stub in enumerate(procedure.inputs):
            if stub.key in input_map:
                procedure.inputs[idx] = input_map[stub.key]
            else:
                logger.warning(
                    "Procedure %d references unknown input key %r", procedure.id, stub.key
                )


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
            inputs: [input1]
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
