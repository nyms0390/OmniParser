"""YAML task template loader for TASK mode.

TASK templates use top-level ``fields``, ``export``, and ``executions`` keys.
The older ``inputs``/``procedures`` shape is intentionally rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from omnitool.gradio.config.enums import (
    AggregateOperation,
    ColumnKind,
    TaskExecutionTool,
    TaskFieldSource,
)

logger = logging.getLogger(__name__)


@dataclass
class TemplateField:
    label: str
    source: TaskFieldSource
    kind: ColumnKind
    description: str = ""
    multiple: bool = False
    expand: bool = False
    clipboard_correction: bool = True


@dataclass
class TaskComputation:
    id: str
    writes: str
    operation: AggregateOperation
    from_fields: list[str]


@dataclass
class TaskExecution:
    id: int
    title: str
    tool: TaskExecutionTool
    system: str
    foreach: str
    uses: list[str]
    writes: list[str]
    steps: str
    resolved_writes: dict[str, TemplateField] = field(default_factory=dict)

    def get_resolved_write(self, key: str) -> TemplateField | None:
        return self.resolved_writes.get(key)


@dataclass
class TaskTemplate:
    name: str
    description: str
    fields: dict[str, TemplateField]
    export: list[str]
    executions: list[TaskExecution]
    computations: list[TaskComputation] = field(default_factory=list)

    def get_field(self, key: str) -> TemplateField | None:
        return self.fields.get(key)


def _require_mapping(data: Any, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping.")
    return data


def _parse_field(key: str, data: Any) -> TemplateField:
    raw = _require_mapping(data, f"field {key!r}")
    raw_source = str(raw.get("source", "")).strip()
    try:
        source = TaskFieldSource(raw_source)
    except ValueError:
        valid = [source.value for source in TaskFieldSource]
        raise ValueError(
            f"field {key!r}: invalid source {raw_source!r}; valid values: {valid}"
        ) from None
    raw_kind = raw.get("kind", ColumnKind.SCALAR.value)
    try:
        kind = ColumnKind(raw_kind)
    except ValueError:
        valid = [k.value for k in ColumnKind]
        raise ValueError(f"field {key!r}: invalid kind {raw_kind!r}; valid values: {valid}") from None
    return TemplateField(
        label=str(raw.get("label") or key),
        source=source,
        kind=kind,
        description=str(raw.get("description", "")),
        multiple=bool(raw.get("multiple", False)),
        expand=bool(raw.get("expand", False)),
        clipboard_correction=bool(raw.get("clipboard_correction", True)),
    )


def _as_string_list(value: Any, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{context} must be a list of field keys.")
    return list(value)


def _validate_refs(keys: list[str], fields: dict[str, TemplateField], context: str) -> None:
    unknown = [key for key in keys if key not in fields]
    if unknown:
        raise ValueError(f"{context}: unknown field reference(s): {unknown!r}")


def _parse_computation(data: Any, fields: dict[str, TemplateField]) -> TaskComputation:
    raw = _require_mapping(data, "computation")
    comp_id = str(raw.get("id") or "").strip()
    writes = str(raw.get("writes") or "").strip()
    from_fields = _as_string_list(raw.get("from"), f"computation {comp_id} from")
    if not comp_id:
        raise ValueError("computation id is required.")
    _validate_refs([writes], fields, f"computation {comp_id} writes")
    _validate_refs(from_fields, fields, f"computation {comp_id} from")
    if not from_fields:
        raise ValueError(f"computation {comp_id}: from must include at least one field key.")
    if fields[writes].source != TaskFieldSource.COMPUTED:
        raise ValueError(f"computation {comp_id}: writes field {writes!r} must be source: computed.")
    file_fields = [key for key in from_fields if fields[key].kind == ColumnKind.FILE]
    if file_fields:
        raise ValueError(f"computation {comp_id}: from cannot include file field(s): {file_fields!r}")
    try:
        operation = AggregateOperation(raw.get("operation"))
    except ValueError:
        valid = [op.value for op in AggregateOperation]
        raise ValueError(
            f"computation {comp_id}: invalid operation {raw.get('operation')!r}; "
            f"valid values: {valid}"
        ) from None
    return TaskComputation(
        id=comp_id,
        writes=writes,
        operation=operation,
        from_fields=from_fields,
    )


def _parse_execution(data: Any, fields: dict[str, TemplateField]) -> TaskExecution:
    raw = _require_mapping(data, "execution")
    exec_id = int(raw.get("id", 0))
    raw_tool = str(raw.get("tool", TaskExecutionTool.CUA.value)).strip()
    try:
        tool = TaskExecutionTool(raw_tool)
    except ValueError:
        valid = [tool.value for tool in TaskExecutionTool]
        raise ValueError(
            f"execution {exec_id}: invalid tool {raw_tool!r}; valid values: {valid}"
        ) from None
    foreach = str(raw.get("foreach") or "").strip()
    uses = _as_string_list(raw.get("uses", []), f"execution {exec_id} uses")
    writes = _as_string_list(raw.get("writes", []), f"execution {exec_id} writes")
    if foreach:
        _validate_refs([foreach], fields, f"execution {exec_id} foreach")
    _validate_refs(uses, fields, f"execution {exec_id} uses")
    _validate_refs(writes, fields, f"execution {exec_id} writes")
    expanding_writes = [key for key in writes if fields[key].expand]
    if len(expanding_writes) > 1:
        raise ValueError(
            f"execution {exec_id}: at most one expand write is allowed; "
            f"got {expanding_writes!r}"
        )
    return TaskExecution(
        id=exec_id,
        title=str(raw.get("title", "")),
        tool=tool,
        system=str(raw.get("system", "")),
        foreach=foreach,
        uses=uses,
        writes=writes,
        steps=str(raw.get("steps", "")),
        resolved_writes={key: fields[key] for key in writes},
    )


def build_execution_task_string(
    execution: TaskExecution,
    template: TaskTemplate,
    row: dict[str, str],
) -> str:
    """Build the per-execution task string sent to the agent."""
    lines = [template.description]
    if row:
        lines.append("\nCurrent row:")
        for key, value in row.items():
            label = template.fields[key].label if key in template.fields else key
            lines.append(f"  - {key}（{label}）: {value}")
    if execution.steps:
        steps = execution.steps.strip()
        for key, value in row.items():
            label = template.fields[key].label if key in template.fields else key
            steps = steps.replace(f"<{key}>（{label}）", value)
            steps = steps.replace(f"<{key}>", value)
        lines.append("\nSteps:")
        lines.append(steps)
    if execution.writes:
        lines.append("\nWritable fields:")
        for key in execution.writes:
            field = template.fields[key]
            lines.append(f"  - {key}（{field.label}）")
    return "\n".join(lines)


def load_task_template(path: str) -> TaskTemplate:
    """Load a YAML task template from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Task template '{path}' must be a mapping.")
    if "procedures" in data or "inputs" in data:
        raise ValueError(
            f"Task template '{path}' uses the retired inputs/procedures format."
        )
    missing = [key for key in ("fields", "export", "executions") if key not in data]
    if missing:
        raise ValueError(f"Task template '{path}' missing required key(s): {missing!r}")

    raw_fields = _require_mapping(data["fields"], "fields")
    fields = {str(key): _parse_field(str(key), value) for key, value in raw_fields.items()}
    export = _as_string_list(data["export"], "export")
    _validate_refs(export, fields, "export")
    computations = [
        _parse_computation(item, fields)
        for item in data.get("computations", []) or []
    ]
    computation_ids = [computation.id for computation in computations]
    duplicate_computation_ids = sorted({
        comp_id for comp_id in computation_ids
        if computation_ids.count(comp_id) > 1
    })
    if duplicate_computation_ids:
        raise ValueError(f"duplicate computation id(s): {duplicate_computation_ids!r}")
    executions = [_parse_execution(item, fields) for item in data["executions"] or []]

    return TaskTemplate(
        name=str(data.get("name", "")),
        description=str(data.get("description", "")),
        fields=fields,
        computations=computations,
        export=export,
        executions=executions,
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
            label = template.name or path.stem
            choices.append((label, str(path)))
        except (yaml.YAMLError, ValueError, OSError) as exc:
            logger.warning("Skipping invalid template %s: %s", path, exc)
    return choices
