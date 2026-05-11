"""TaskRunner orchestrates TASK template executions over a row dataframe."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Protocol

from omnitool.gradio.config import (
    TaskExecution,
    TaskTemplate,
    build_execution_task_string,
)

logger = logging.getLogger(__name__)


class RunnableAgent(Protocol):
    def run(self) -> Iterator[Dict[str, Any]]: ...


AgentFactory = Callable[[TaskExecution, str], RunnableAgent]


class TaskDataframe:
    """Row-oriented dataframe assembled across one task run."""

    def __init__(self, columns: List[str], labels: dict[str, str] | None = None) -> None:
        self.columns = list(columns)
        self.labels = dict(labels or {})
        self.rows: List[Dict[str, str]] = []

    def set_cell(self, row_idx: int, key: str, value: str) -> None:
        if not 0 <= row_idx < len(self.rows):
            raise IndexError(
                f"set_cell row_idx={row_idx} out of range (have {len(self.rows)} rows)"
            )
        self.rows[row_idx][key] = value

    def expand_row(self, row_idx: int, key: str, values: List[str]) -> None:
        if not 0 <= row_idx < len(self.rows):
            raise IndexError(
                f"expand_row row_idx={row_idx} out of range (have {len(self.rows)} rows)"
            )
        original = self.rows[row_idx]
        self.rows[row_idx:row_idx + 1] = [{**original, key: value} for value in values]

    def to_csv(self) -> str:
        buf = io.StringIO()
        headers = [self.labels.get(key, key) for key in self.columns]
        writer = csv.DictWriter(
            buf,
            fieldnames=self.columns,
            extrasaction="ignore",
        )
        writer.writerow(dict(zip(self.columns, headers)))
        for row in self.rows:
            writer.writerow(row)
        return buf.getvalue()


def _join(values: List[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values)


class TaskRunner:
    """Drive a task end-to-end via per-row agent executions."""

    def __init__(
        self,
        template: TaskTemplate,
        agent_factory: AgentFactory,
        save_folder: Path,
        user_values: dict[str, Any] | None = None,
    ) -> None:
        self.template = template
        self.agent_factory = agent_factory
        self.save_folder = Path(save_folder)
        self.user_values = user_values or {}
        self.dataframe = TaskDataframe(
            columns=template.export,
            labels={key: template.fields[key].label for key in template.export},
        )
        self._allow_empty_seed_run = not any(
            field.source == "user" for field in template.fields.values()
        )
        self.dataframe.rows = self._seed_rows()

    def _seed_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = [{}]
        for key, field in self.template.fields.items():
            if field.source != "user" or key not in self.user_values:
                continue
            value = self.user_values[key]
            if isinstance(value, list) and field.expand:
                rows = [
                    {**row, key: str(item)}
                    for row in rows
                    for item in value
                ]
            elif isinstance(value, list):
                joined = _join([str(item) for item in value])
                rows = [{**row, key: joined} for row in rows]
            else:
                rows = [{**row, key: str(value)} for row in rows]
        return rows

    def run_task(self) -> Generator[Dict[str, Any], None, None]:
        for execution in self.template.executions:
            original_indices = [
                idx for idx, row in enumerate(self.dataframe.rows)
                if execution.foreach in row and row.get(execution.foreach, "") != ""
            ]
            if (
                not original_indices
                and self._allow_empty_seed_run
                and self.dataframe.rows == [{}]
            ):
                original_indices = [0]
            cursor_offset = 0
            all_ok = True
            for original_idx in original_indices:
                row_idx = original_idx + cursor_offset
                before = len(self.dataframe.rows)
                success = yield from self._run_once(execution, row_idx)
                self._apply_computations()
                cursor_offset += len(self.dataframe.rows) - before
                if not success:
                    all_ok = False
                    break
            if not all_ok:
                yield {
                    "type": "task_complete",
                    "success": False,
                    "csv_path": None,
                    "rows": self.dataframe.rows,
                }
                return
            yield {"type": "execution_complete", "execution_id": execution.id}

        csv_path = self.save_folder / "task_result.csv"
        csv_written = False
        try:
            csv_path.write_text(self.dataframe.to_csv(), encoding="utf-8")
            csv_written = True
        except OSError as exc:
            logger.warning("Failed to write %s: %s", csv_path, exc)

        yield {
            "type": "task_complete",
            "success": True,
            "csv_path": str(csv_path) if csv_written else None,
            "rows": self.dataframe.rows,
        }

    def run_once(
        self, execution: TaskExecution, row_idx: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        if not 0 <= row_idx < len(self.dataframe.rows):
            yield {
                "type": "error",
                "message": (
                    f"run_once: row_idx={row_idx} out of range "
                    f"(have {len(self.dataframe.rows)} rows)"
                ),
            }
            return
        task_string = build_execution_task_string(
            execution, self.template, self.dataframe.rows[row_idx]
        )
        agent = self.agent_factory(execution, task_string)
        yield from agent.run()

    def _run_once(
        self, execution: TaskExecution, row_idx: int,
    ) -> Generator[Dict[str, Any], None, bool]:
        task_string = build_execution_task_string(
            execution, self.template, self.dataframe.rows[row_idx]
        )
        agent = self.agent_factory(execution, task_string)

        saw_complete = False
        saw_error = False
        for event in agent.run():
            evt_type = event.get("type")
            if evt_type == "complete":
                self._merge_facts(event.get("facts", {}), execution, row_idx)
                saw_complete = True
                continue
            if evt_type == "error":
                saw_error = True
            yield event
        return saw_complete and not saw_error

    def _merge_facts(
        self,
        facts: Dict[str, List[str]],
        execution: TaskExecution,
        row_idx: int,
    ) -> None:
        scalar_items: List[tuple[str, List[str]]] = []
        expand_items: List[tuple[str, List[str]]] = []
        declared = set(execution.writes)
        for key, values in facts.items():
            if key not in declared:
                logger.warning(
                    "Fact %r not declared in execution %s writes; skipping",
                    key, execution.id,
                )
                continue
            field = self.template.fields.get(key)
            if field is None:
                logger.warning("Unknown field key %r; skipping", key)
                continue
            (expand_items if field.expand else scalar_items).append((key, values))

        for key, values in scalar_items:
            if values:
                self.dataframe.set_cell(row_idx, key, _join(values))
        for key, values in expand_items:
            self.dataframe.expand_row(row_idx, key, values)

    def _apply_computations(self) -> None:
        for row in self.dataframe.rows:
            for computation in self.template.computations:
                raw = row.get(computation.from_field)
                if raw is None:
                    continue
                values = [item.strip() for item in raw.split(",") if item.strip()]
                try:
                    result = computation.operation.apply(values)
                except (NotImplementedError, ValueError) as exc:
                    logger.warning(
                        "Computation %s failed for field %s: %s",
                        computation.id, computation.from_field, exc,
                    )
                    continue
                row[computation.writes] = _join(result) if isinstance(result, list) else result


__all__ = ["TaskDataframe", "TaskRunner", "RunnableAgent", "AgentFactory"]
