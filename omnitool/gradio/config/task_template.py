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
from typing import Dict, List, Optional

import yaml

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
    """A single output field produced by a procedure."""

    key: str
    description: str = ""
    format: str = ""

    @classmethod
    def from_dict(cls, data) -> "TaskOutput":
        if isinstance(data, str):
            return cls(key=data)
        return cls(
            key=data["key"],
            description=data.get("description", ""),
            format=data.get("format", ""),
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
    inputs: List[str] = field(default_factory=list)  # key references
    outputs: List[TaskOutput] = field(default_factory=list)
    executions: List[TaskExecution] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _substitute_inputs(self, text: str, inputs: List[TaskInput]) -> str:
        """Replace ``<key>`` placeholders in *text* with resolved input values."""
        for inp in inputs:
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

    def to_task_string(self, inputs: List[TaskInput]) -> str:
        """Build a task string suitable for ``_init_checklist_from_template``.

        The string contains:
        - Procedure description
        - Inputs summary (key: value or N/A) for resolved inputs
        - Numbered steps from CUA executions (with input values substituted)
        - Outputs to capture block for any ``<key>`` output references in steps

        The numbered steps are embedded so that
        ``Checklist.from_user_text()`` can parse them into checklist items.

        Args:
            inputs: Resolved :class:`TaskInput` objects for this procedure,
                obtained via :meth:`TaskTemplate.resolve_inputs`.
        """
        lines = [self.description]

        if inputs:
            lines.append("\nInputs:")
            for inp in inputs:
                val = str(inp.value) if inp.value is not None else "N/A"
                lines.append(f"  - {inp.key}: {val}")

        steps_raw = self._cua_steps()
        if steps_raw:
            steps_substituted = self._substitute_inputs(steps_raw, inputs)
            lines.append("\nSteps:")
            lines.append(steps_substituted)

            referenced = self._referenced_outputs(steps_raw)
            if referenced:
                lines.append("\nOutputs to capture:")
                for out in referenced:
                    entry = f"  - {out.key}"
                    if out.description:
                        entry += f": {out.description}"
                    if out.format:
                        entry += f" (format: {out.format})"
                    lines.append(entry)

        return "\n".join(lines)

    def to_extract_fields(self) -> Dict[str, str]:
        """Return ``{output.key: output.description}`` for all outputs."""
        return {out.key: out.description for out in self.outputs}

    # ------------------------------------------------------------------
    # Class-level constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "TaskProcedure":
        if "ID" not in data and "id" not in data:
            raise ValueError("Procedure dict missing 'ID' field.")
        proc_id = data.get("ID", data.get("id"))
        raw_inputs = data.get("inputs", [])
        # Accept both plain strings and dicts with a "key" field.
        input_keys = [
            i if isinstance(i, str) else i["key"] for i in raw_inputs
        ]
        return cls(
            id=int(proc_id),
            description=data.get("description", ""),
            inputs=input_keys,
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

    def resolve_inputs(self, procedure: TaskProcedure) -> List[TaskInput]:
        """Return :class:`TaskInput` objects for keys listed in *procedure*.

        Keys not found in the top-level inputs produce a warning and are skipped.
        """
        input_map = {i.key: i for i in self.inputs}
        resolved = []
        for k in procedure.inputs:
            if k in input_map:
                resolved.append(input_map[k])
            else:
                logger.warning(
                    "Procedure %d references unknown input key %r", procedure.id, k
                )
        return resolved


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
