"""YAML task template loader for TASK mode.

Parses a YAML file that is a top-level list of procedure definitions.
Only the first procedure is used for execution.

YAML schema (per procedure)::

    - ID: 1
      description: "Procedure description"
      inputs:
        - key: input1
          value: 12345678       # pre-filled by user
          description: "..."
          format: 8 number digits
          required: true
      outputs:
        - key: output1
          description: "..."
          format: string
      executions:
        - type: cua
          system: EPA
          steps: |
            1. Step one.
            2. Step two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class TaskInput:
    """A single input parameter for a procedure."""

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
    inputs: List[TaskInput] = field(default_factory=list)
    outputs: List[TaskOutput] = field(default_factory=list)
    executions: List[TaskExecution] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _substitute_inputs(self, text: str) -> str:
        """Replace ``<key>`` placeholders in *text* with their input values."""
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

    def to_task_string(self) -> str:
        """Build a task string suitable for ``_init_checklist_from_template``.

        The string contains:
        - Procedure description
        - Inputs summary (key: value or N/A)
        - Numbered steps from CUA executions (with input values substituted)

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
            steps_substituted = self._substitute_inputs(steps_raw)
            lines.append("\nSteps:")
            lines.append(steps_substituted)

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
        proc_id = data.get("ID") or data.get("id")
        return cls(
            id=int(proc_id),
            description=data.get("description", ""),
            inputs=[TaskInput.from_dict(i) for i in data.get("inputs", [])],
            outputs=[TaskOutput.from_dict(o) for o in data.get("outputs", [])],
            executions=[TaskExecution.from_dict(e) for e in data.get("executions", [])],
        )


def load_task_template(path: str) -> TaskProcedure:
    """Load a YAML task template and return the first procedure.

    Accepts two YAML formats:

    1. Top-level list::

        - ID: 1
          description: "..."
          ...

    2. Procedures wrapped under a ``procedure`` key::

        procedure:
          - ID: 1
            description: "..."
            ...

    Only the first procedure is loaded and returned.

    Args:
        path: Filesystem path to the ``.yaml`` / ``.yml`` file.

    Returns:
        :class:`TaskProcedure` for the first item in the procedure list.

    Raises:
        ValueError: If the file is empty, not a recognised format, or has
            no procedures.
        yaml.YAMLError: If the file cannot be parsed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if isinstance(data, list):
        procedures = data
    elif isinstance(data, dict) and "procedures" in data:
        procedures = data["procedures"]
        if not isinstance(procedures, list):
            raise ValueError(
                f"Task template '{path}': 'procedures' key must contain a list."
            )
    else:
        raise ValueError(
            f"Task template '{path}' must be either a top-level YAML list or a "
            "mapping with a 'procedure' key."
        )

    if not procedures:
        raise ValueError(f"Task template '{path}' contains no procedures.")

    return TaskProcedure.from_dict(procedures[0])
