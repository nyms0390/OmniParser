"""ProcedureRunner — orchestrates multiple agent runs for one procedure.

The runner maintains a row-oriented dataframe seeded with template scalars.
Every execution iterates a snapshot of the current rows; each iteration is
one agent run. Per-output write semantics are driven by the procedure
schema's ``explode`` flag:

- ``explode=True``  → the source row is replaced by N rows (one per value).
- ``explode=False`` → values are joined into a single cell on the source row.

A chain like ``user_id → accounts → transactions → balance`` falls out
naturally: each explode step expands the dataframe; subsequent executions
iterate the larger row set.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Protocol

from omnitool.gradio.config import (
    TaskExecution,
    TaskProcedure,
    TaskTemplate,
    build_execution_task_string,
)

logger = logging.getLogger(__name__)


class RunnableAgent(Protocol):
    """The minimal contract the runner needs from an agent.

    The factory must return an object whose ``run()`` is a generator of event
    dicts. The runner expects exactly one terminal ``complete`` event carrying
    ``facts: Dict[str, List[str]]``; any ``error`` event aborts the procedure.
    """

    def run(self) -> Iterator[Dict[str, Any]]: ...


AgentFactory = Callable[[TaskExecution, str], RunnableAgent]


class ProcedureDataframe:
    """Row-oriented dataframe assembled across one procedure run.

    Columns are declared from the procedure schema. ``explode_row`` replaces
    one row with N rows (one per value); ``set_cell`` writes a value into an
    existing row. Rows are plain ``Dict[str, str]``; extra keys (e.g.
    template inputs riding along as substitution context) are dropped by
    ``to_csv`` because ``DictWriter`` is configured with ``extrasaction="ignore"``.
    """

    def __init__(self, columns: List[str]) -> None:
        self.columns: List[str] = list(columns)
        self.rows: List[Dict[str, str]] = []

    def set_cell(self, row_idx: int, key: str, value: str) -> None:
        """Write *value* into the cell at ``(row_idx, key)``."""
        if not 0 <= row_idx < len(self.rows):
            raise IndexError(
                f"set_cell row_idx={row_idx} out of range (have {len(self.rows)} rows)"
            )
        self.rows[row_idx][key] = value

    def explode_row(self, row_idx: int, key: str, values: List[str]) -> None:
        """Replace the row at *row_idx* with one copy per value in *values*.

        Each new row inherits the original's keys and adds ``{key: value}``.
        An empty *values* removes the source row.
        """
        if not 0 <= row_idx < len(self.rows):
            raise IndexError(
                f"explode_row row_idx={row_idx} out of range (have {len(self.rows)} rows)"
            )
        original = self.rows[row_idx]
        new_rows = [{**original, key: v} for v in values]
        self.rows[row_idx:row_idx + 1] = new_rows

    def to_csv(self) -> str:
        """Render to CSV text using ``self.columns`` as the header order."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.columns, extrasaction="ignore")
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row)
        return buf.getvalue()


def _join(values: List[str]) -> str:
    """Collapse a fact list into a single cell value."""
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values)


class ProcedureRunner:
    """Drive one procedure end-to-end via a unified per-row agent loop."""

    def __init__(
        self,
        procedure: TaskProcedure,
        template: TaskTemplate,
        agent_factory: AgentFactory,
        save_folder: Path,
    ) -> None:
        self.procedure = procedure
        self.template = template
        self.agent_factory = agent_factory
        self.save_folder = Path(save_folder)
        self.dataframe = ProcedureDataframe(
            columns=[o.key for o in procedure.outputs],
        )
        seed = {ti.key: ti.value for ti in template.inputs if ti.value is not None}
        self.dataframe.rows = [seed]

    def run_execution(
        self, execution: TaskExecution,
    ) -> Generator[Dict[str, Any], None, bool]:
        """Run one execution across the current dataframe rows.

        Iterates the rows that existed before the execution started; ``cursor``
        advances past the row just processed plus any new rows it produced via
        explode. An empty-explode (delta=-1) leaves cursor in place: the next
        original row has shifted into the freed slot.

        Yields agent events (forwarded from ``_run_once``) plus a terminal
        ``execution_complete`` on success. Returns ``True`` if every sub-run
        succeeded, ``False`` if any agent run errored. Does not write CSV.
        """
        original_count = len(self.dataframe.rows)
        cursor = 0
        for _ in range(original_count):
            before = len(self.dataframe.rows)
            success = yield from self._run_once(execution, cursor)
            delta = len(self.dataframe.rows) - before
            if delta >= 0:
                cursor += 1 + delta
            if not success:
                return False
        yield {"type": "execution_complete", "execution_id": execution.id}
        return True

    def run(self) -> Generator[Dict[str, Any], None, None]:
        """Run every execution per-row; emit events; write CSV at the end."""
        for execution in self.procedure.executions:
            success = yield from self.run_execution(execution)
            if not success:
                yield {
                    "type": "procedure_complete",
                    "success": False,
                    "csv_path": None,
                    "rows": self.dataframe.rows,
                }
                return

        csv_path = self.save_folder / "procedure_result.csv"
        csv_written = False
        try:
            csv_path.write_text(self.dataframe.to_csv(), encoding="utf-8")
            csv_written = True
        except OSError as exc:
            logger.warning("Failed to write %s: %s", csv_path, exc)

        yield {
            "type": "procedure_complete",
            "success": True,
            "csv_path": str(csv_path) if csv_written else None,
            "rows": self.dataframe.rows,
        }

    def _run_once(
        self, execution: TaskExecution, row_idx: int,
    ) -> Generator[Dict[str, Any], None, bool]:
        """Run one agent invocation against ``rows[row_idx]``.

        Returns True on success (saw a ``complete`` event and no ``error``).
        """
        row = self.dataframe.rows[row_idx]
        task_string = build_execution_task_string(execution, self.procedure, row)
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
        """Write committed facts back to the dataframe.

        Only keys declared in ``execution.outputs`` are honoured. This
        enforces the "≤1 explode per execution" invariant at runtime —
        ``TaskProcedure.from_dict`` already rejects two-explode declarations,
        so the second pass below replaces the row at most once.

        Two passes: scalars first (so the source row carries them), then
        explode (so the new rows inherit the scalars). A scalar with no
        committed values is skipped, not blanked, so it cannot overwrite
        a value written by an earlier execution.
        """
        declared = {eo.key for eo in execution.outputs}
        scalar_items: List[tuple[str, List[str]]] = []
        explode_items: List[tuple[str, List[str]]] = []
        for key, values in facts.items():
            if key not in declared:
                logger.warning(
                    "Fact %r not declared in execution %s outputs — skipping",
                    key, execution.id,
                )
                continue
            schema_out = self.procedure.get_output(key)
            if schema_out is None:
                logger.warning("Unknown output key %r — not in procedure schema", key)
                continue
            (explode_items if schema_out.explode else scalar_items).append((key, values))

        for key, values in scalar_items:
            if not values:
                continue
            self.dataframe.set_cell(row_idx, key, _join(values))

        for key, values in explode_items:
            self.dataframe.explode_row(row_idx, key, values)


__all__ = ["ProcedureDataframe", "ProcedureRunner", "RunnableAgent", "AgentFactory"]
