"""Tests for ProcedureDataframe and ProcedureRunner (unified per-row loop)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from omnitool.gradio.config.enums import ColumnKind
from omnitool.gradio.config.task_template import (
    ExecutionOutput,
    TaskExecution,
    TaskInput,
    TaskOutput,
    TaskProcedure,
    TaskTemplate,
)
from omnitool.gradio.core.procedure_runner import ProcedureDataframe, ProcedureRunner


# ---------------------------------------------------------------------------
# ProcedureDataframe
# ---------------------------------------------------------------------------

class TestProcedureDataframe:
    def test_set_cell_writes_into_existing_row(self):
        df = ProcedureDataframe(columns=["a", "b"])
        df.rows = [{"a": "x"}]
        df.set_cell(0, "b", "y")
        assert df.rows[0] == {"a": "x", "b": "y"}

    def test_set_cell_out_of_range_raises(self):
        df = ProcedureDataframe(columns=["a"])
        try:
            df.set_cell(0, "a", "x")
        except IndexError as exc:
            assert "out of range" in str(exc)
        else:
            raise AssertionError("expected IndexError")

    def test_explode_row_replaces_with_n_rows(self):
        df = ProcedureDataframe(columns=["seed", "k"])
        df.rows = [{"seed": "s"}]
        df.explode_row(0, "k", ["v1", "v2", "v3"])
        assert df.rows == [
            {"seed": "s", "k": "v1"},
            {"seed": "s", "k": "v2"},
            {"seed": "s", "k": "v3"},
        ]

    def test_explode_row_empty_values_removes_row(self):
        df = ProcedureDataframe(columns=["k"])
        df.rows = [{"k": "x"}, {"k": "y"}]
        df.explode_row(0, "k", [])
        assert df.rows == [{"k": "y"}]

    def test_explode_row_inherits_existing_keys(self):
        df = ProcedureDataframe(columns=["user_id", "account"])
        df.rows = [{"user_id": "U1"}]
        df.explode_row(0, "account", ["A1", "A2"])
        assert all(r["user_id"] == "U1" for r in df.rows)

    def test_to_csv_uses_declared_column_order(self):
        df = ProcedureDataframe(columns=["a", "b"])
        df.rows = [{"a": "1", "b": "2", "extra": "ignored"}]
        lines = df.to_csv().strip().splitlines()
        assert lines[0] == "a,b"
        assert lines[1] == "1,2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _example_template() -> TaskTemplate:
    """The 2-execution worked example: user_id → account_id (explode) → balance/status."""
    proc = TaskProcedure(
        id=1,
        description="Pull accounts and enrich each.",
        outputs=[
            TaskOutput(key="account_id", kind=ColumnKind.ROW, explode=True),
            TaskOutput(key="balance", kind=ColumnKind.SCALAR),
            TaskOutput(key="status", kind=ColumnKind.SCALAR),
        ],
        executions=[
            TaskExecution(
                id=1, type="cua", system="EPA",
                inputs=["user_id"],
                outputs=[ExecutionOutput(key="account_id")],
                steps="Open <user_id> account list.",
            ),
            TaskExecution(
                id=2, type="cua", system="EPA",
                inputs=["account_id"],
                outputs=[ExecutionOutput(key="balance"), ExecutionOutput(key="status")],
                steps="Open profile for <account_id>.",
            ),
        ],
    )
    return TaskTemplate(
        inputs=[TaskInput(key="user_id", value="12345")],
        procedures=[proc],
    )


def _scripted_factory(
    completes: Iterable[Any],
    *,
    extra_events: Iterable[List[Dict[str, Any]]] = (),
):
    """Factory yielding scripted complete events.

    *completes* entries: ``Dict[str, List[str]]`` for ``facts`` payload, OR
    the sentinels ``"error"`` (yields an error event then no complete) and
    ``"crash"`` (yields no complete at all).
    """
    completes_iter = iter(completes)
    extras_iter = iter(list(extra_events) + [[]] * 100)
    calls: List[Dict[str, Any]] = []

    def factory(execution: TaskExecution, task_string: str):
        directive = next(completes_iter)
        extras = next(extras_iter)
        calls.append({
            "execution_id": execution.id,
            "task_string": task_string,
            "directive": directive,
        })

        class _Agent:
            def run(self_inner):
                for evt in extras:
                    yield evt
                if directive == "error":
                    yield {"type": "error", "message": "boom"}
                    return
                if directive == "crash":
                    return
                yield {
                    "type": "complete", "facts": directive,
                    "success": True, "message": "",
                    "total_steps": 1, "total_tokens": 0, "total_cost": "$0",
                }

        return _Agent()

    factory.calls = calls
    return factory


# ---------------------------------------------------------------------------
# ProcedureRunner — happy path
# ---------------------------------------------------------------------------

class TestProcedureRunnerHappyPath:
    def test_two_execution_example_builds_expected_dataframe(self, tmp_path):
        template = _example_template()
        procedure = template.procedures[0]
        factory = _scripted_factory([
            {"account_id": ["A001", "A002", "A003"]},
            {"balance": ["10.00"], "status": ["active"]},
            {"balance": ["20.00"], "status": ["closed"]},
            {"balance": ["30.00"], "status": ["active"]},
        ])

        runner = ProcedureRunner(procedure, template, factory, tmp_path)
        events = list(runner.run())

        assert runner.dataframe.rows == [
            {"user_id": "12345", "account_id": "A001", "balance": "10.00", "status": "active"},
            {"user_id": "12345", "account_id": "A002", "balance": "20.00", "status": "closed"},
            {"user_id": "12345", "account_id": "A003", "balance": "30.00", "status": "active"},
        ]

        csv_path = tmp_path / "procedure_1_result.csv"
        assert csv_path.exists()
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "account_id,balance,status"  # user_id excluded — not in schema
        assert "A002,20.00,closed" in lines[2]

        types = [e["type"] for e in events]
        assert types.count("execution_complete") == 2
        assert types[-1] == "procedure_complete"
        assert events[-1]["success"] is True

    def test_seed_row_carries_template_scalars_into_substitution(self, tmp_path):
        template = _example_template()
        procedure = template.procedures[0]
        factory = _scripted_factory([
            {"account_id": ["A001"]},
            {"balance": ["1"], "status": ["x"]},
        ])

        runner = ProcedureRunner(procedure, template, factory, tmp_path)
        list(runner.run())
        assert "12345" in factory.calls[0]["task_string"]
        assert "A001" in factory.calls[1]["task_string"]

    def test_chained_explode_user_to_accounts_to_transactions(self, tmp_path):
        """user_id → accounts (explode) → transactions (explode) → balance."""
        proc = TaskProcedure(
            id=2,
            description="chain",
            outputs=[
                TaskOutput(key="accounts", kind=ColumnKind.ROW, explode=True),
                TaskOutput(key="transactions", kind=ColumnKind.ROW, explode=True),
                TaskOutput(key="balance", kind=ColumnKind.SCALAR),
            ],
            executions=[
                TaskExecution(id=1, type="cua", system="EPA",
                              inputs=["user_id"],
                              outputs=[ExecutionOutput(key="accounts")],
                              steps="<user_id>"),
                TaskExecution(id=2, type="cua", system="EPA",
                              inputs=["accounts"],
                              outputs=[ExecutionOutput(key="transactions")],
                              steps="<accounts>"),
                TaskExecution(id=3, type="cua", system="EPA",
                              inputs=["transactions"],
                              outputs=[ExecutionOutput(key="balance")],
                              steps="<transactions>"),
            ],
        )
        template = TaskTemplate(
            inputs=[TaskInput(key="user_id", value="U1")],
            procedures=[proc],
        )
        factory = _scripted_factory([
            {"accounts": ["A1", "A2"]},
            {"transactions": ["T1", "T2"]},
            {"transactions": ["T3"]},
            {"balance": ["10"]},
            {"balance": ["20"]},
            {"balance": ["30"]},
        ])
        runner = ProcedureRunner(proc, template, factory, tmp_path)
        list(runner.run())

        assert len(runner.dataframe.rows) == 3
        observed = [(r["accounts"], r["transactions"], r["balance"]) for r in runner.dataframe.rows]
        assert observed == [("A1", "T1", "10"), ("A1", "T2", "20"), ("A2", "T3", "30")]


# ---------------------------------------------------------------------------
# ProcedureRunner — write semantics
# ---------------------------------------------------------------------------

class TestMergeFactsSemantics:
    def test_kind_row_explode_false_joins_into_one_cell(self, tmp_path):
        """A non-exploding multi-value output collapses to a comma-joined cell."""
        proc = TaskProcedure(
            id=1, description="",
            outputs=[TaskOutput(key="tags", kind=ColumnKind.ROW, explode=False)],
            executions=[TaskExecution(id=1, type="cua", system="EPA",
                                      inputs=[], outputs=[ExecutionOutput(key="tags")],
                                      steps="")],
        )
        template = TaskTemplate(inputs=[], procedures=[proc])
        factory = _scripted_factory([{"tags": ["red", "green", "blue"]}])

        runner = ProcedureRunner(proc, template, factory, tmp_path)
        list(runner.run())
        assert runner.dataframe.rows == [{"tags": "red, green, blue"}]

    def test_scalar_with_one_value_writes_that_value(self, tmp_path):
        proc = TaskProcedure(
            id=1, description="",
            outputs=[TaskOutput(key="name", kind=ColumnKind.SCALAR)],
            executions=[TaskExecution(id=1, type="cua", system="EPA",
                                      inputs=[], outputs=[ExecutionOutput(key="name")],
                                      steps="")],
        )
        template = TaskTemplate(inputs=[], procedures=[proc])
        factory = _scripted_factory([{"name": ["Alice"]}])

        runner = ProcedureRunner(proc, template, factory, tmp_path)
        list(runner.run())
        assert runner.dataframe.rows == [{"name": "Alice"}]

    def test_scalar_lands_on_seed_row_before_explode(self, tmp_path):
        """In one execution: scalar set first, then explode — new rows inherit it."""
        proc = TaskProcedure(
            id=1, description="",
            outputs=[
                TaskOutput(key="name", kind=ColumnKind.SCALAR),
                TaskOutput(key="account", kind=ColumnKind.ROW, explode=True),
            ],
            executions=[TaskExecution(
                id=1, type="cua", system="EPA",
                inputs=[],
                outputs=[ExecutionOutput(key="name"), ExecutionOutput(key="account")],
                steps="",
            )],
        )
        template = TaskTemplate(inputs=[], procedures=[proc])
        factory = _scripted_factory([{"name": ["Alice"], "account": ["A1", "A2"]}])

        runner = ProcedureRunner(proc, template, factory, tmp_path)
        list(runner.run())
        assert runner.dataframe.rows == [
            {"name": "Alice", "account": "A1"},
            {"name": "Alice", "account": "A2"},
        ]

    def test_unknown_output_key_logged_and_skipped(self, tmp_path, caplog):
        proc = TaskProcedure(
            id=1, description="",
            outputs=[TaskOutput(key="known", kind=ColumnKind.SCALAR)],
            executions=[TaskExecution(id=1, type="cua", system="EPA",
                                      inputs=[], outputs=[ExecutionOutput(key="known")],
                                      steps="")],
        )
        template = TaskTemplate(inputs=[], procedures=[proc])
        factory = _scripted_factory([{"known": ["v"], "stray": ["x"]}])

        runner = ProcedureRunner(proc, template, factory, tmp_path)
        with caplog.at_level("WARNING"):
            list(runner.run())
        assert runner.dataframe.rows == [{"known": "v"}]
        assert any("stray" in r.message for r in caplog.records)

    def test_empty_scalar_does_not_blank_existing_cell(self, tmp_path):
        """A scalar fact with empty values is skipped, not written as ""."""
        proc = TaskProcedure(
            id=1, description="",
            outputs=[TaskOutput(key="status", kind=ColumnKind.SCALAR)],
            executions=[
                TaskExecution(id=1, type="cua", system="EPA",
                              inputs=[], outputs=[ExecutionOutput(key="status")],
                              steps=""),
                TaskExecution(id=2, type="cua", system="EPA",
                              inputs=[], outputs=[ExecutionOutput(key="status")],
                              steps=""),
            ],
        )
        template = TaskTemplate(inputs=[], procedures=[proc])
        factory = _scripted_factory([
            {"status": ["active"]},
            {"status": []},  # empty list must NOT blank the prior value
        ])
        runner = ProcedureRunner(proc, template, factory, tmp_path)
        list(runner.run())
        assert runner.dataframe.rows == [{"status": "active"}]

    def test_undeclared_fact_is_skipped(self, tmp_path, caplog):
        """A fact key not declared in execution.outputs is dropped at runtime.

        This guards the "≤1 explode per execution" invariant against an agent
        that commits an undeclared explode fact alongside the declared one.
        """
        proc = TaskProcedure(
            id=1, description="",
            outputs=[
                TaskOutput(key="primary", kind=ColumnKind.ROW, explode=True),
                TaskOutput(key="other", kind=ColumnKind.ROW, explode=True),
            ],
            executions=[TaskExecution(
                id=1, type="cua", system="EPA",
                inputs=[],
                outputs=[ExecutionOutput(key="primary")],  # only primary
                steps="",
            )],
        )
        template = TaskTemplate(inputs=[], procedures=[proc])
        factory = _scripted_factory([
            {"primary": ["A", "B"], "other": ["C", "D", "E"]},
        ])
        runner = ProcedureRunner(proc, template, factory, tmp_path)
        with caplog.at_level("WARNING"):
            list(runner.run())
        # Only `primary` exploded → 2 rows. Without the runtime filter, `other`
        # would have produced a cartesian product of 6 rows.
        assert len(runner.dataframe.rows) == 2
        assert all("other" not in r for r in runner.dataframe.rows)
        assert any("other" in r.message for r in caplog.records)

    def test_empty_explode_removes_seed_row(self, tmp_path):
        """Agent returning zero row values ⇒ downstream rows = 0; CSV header only."""
        template = _example_template()
        procedure = template.procedures[0]
        factory = _scripted_factory([{"account_id": []}])

        runner = ProcedureRunner(procedure, template, factory, tmp_path)
        list(runner.run())
        assert runner.dataframe.rows == []
        # Execution 2 ran zero times because no rows
        assert [c["execution_id"] for c in factory.calls] == [1]


# ---------------------------------------------------------------------------
# ProcedureRunner — failure modes
# ---------------------------------------------------------------------------

class TestProcedureRunnerFailures:
    def test_error_event_aborts_procedure(self, tmp_path):
        template = _example_template()
        procedure = template.procedures[0]
        factory = _scripted_factory([
            {"account_id": ["A1", "A2", "A3"]},
            {"balance": ["10"], "status": ["active"]},
            "error",  # row 1 of execution 2 fails
        ])

        runner = ProcedureRunner(procedure, template, factory, tmp_path)
        events = list(runner.run())

        # Only first 2 + error agent calls happened — row 3 never run
        assert [c["execution_id"] for c in factory.calls] == [1, 2, 2]
        proc_complete = [e for e in events if e["type"] == "procedure_complete"][0]
        assert proc_complete["success"] is False
        assert proc_complete["csv_path"] is None
        assert not (tmp_path / "procedure_1_result.csv").exists()

    def test_agent_returns_without_complete_treated_as_failure(self, tmp_path):
        template = _example_template()
        procedure = template.procedures[0]
        factory = _scripted_factory([
            {"account_id": ["A1"]},
            "crash",
        ])

        runner = ProcedureRunner(procedure, template, factory, tmp_path)
        events = list(runner.run())
        proc_complete = [e for e in events if e["type"] == "procedure_complete"][0]
        assert proc_complete["success"] is False
        assert not (tmp_path / "procedure_1_result.csv").exists()

    def test_runner_drops_complete_events_from_agent(self, tmp_path):
        template = _example_template()
        procedure = template.procedures[0]
        factory = _scripted_factory([
            {"account_id": ["A1"]},
            {"balance": ["1"], "status": ["x"]},
        ])

        runner = ProcedureRunner(procedure, template, factory, tmp_path)
        events = list(runner.run())
        assert all(e["type"] != "complete" for e in events)

    def test_runner_forwards_non_complete_events(self, tmp_path):
        template = _example_template()
        procedure = template.procedures[0]
        factory = _scripted_factory(
            completes=[{"account_id": ["A1"]}, {"balance": ["1"], "status": ["x"]}],
            extra_events=[
                [{"type": "step", "step_num": 1}, {"type": "thinking", "response_text": "..."}],
                [{"type": "step", "step_num": 1}],
            ],
        )

        runner = ProcedureRunner(procedure, template, factory, tmp_path)
        events = list(runner.run())
        forwarded = [e for e in events if e["type"] in ("step", "thinking")]
        assert len(forwarded) == 3
