"""Tests for TaskDataframe and TaskRunner (unified per-row loop)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from omnitool.gradio.config.enums import ColumnKind
from omnitool.gradio.config.task_template import (
    TaskExecution,
    TaskTemplate,
    TemplateField,
)
from omnitool.gradio.core.task_runner import TaskDataframe, TaskRunner


# ---------------------------------------------------------------------------
# TaskDataframe
# ---------------------------------------------------------------------------

class TestTaskDataframe:
    def test_set_cell_writes_into_existing_row(self):
        df = TaskDataframe(columns=["a", "b"])
        df.rows = [{"a": "x"}]
        df.set_cell(0, "b", "y")
        assert df.rows[0] == {"a": "x", "b": "y"}

    def test_set_cell_out_of_range_raises(self):
        df = TaskDataframe(columns=["a"])
        try:
            df.set_cell(0, "a", "x")
        except IndexError as exc:
            assert "out of range" in str(exc)
        else:
            raise AssertionError("expected IndexError")

    def test_expand_row_replaces_with_n_rows(self):
        df = TaskDataframe(columns=["seed", "k"])
        df.rows = [{"seed": "s"}]
        df.expand_row(0, "k", ["v1", "v2", "v3"])
        assert df.rows == [
            {"seed": "s", "k": "v1"},
            {"seed": "s", "k": "v2"},
            {"seed": "s", "k": "v3"},
        ]

    def test_expand_row_empty_values_removes_row(self):
        df = TaskDataframe(columns=["k"])
        df.rows = [{"k": "x"}, {"k": "y"}]
        df.expand_row(0, "k", [])
        assert df.rows == [{"k": "y"}]

    def test_expand_row_inherits_existing_keys(self):
        df = TaskDataframe(columns=["user_id", "account"])
        df.rows = [{"user_id": "U1"}]
        df.expand_row(0, "account", ["A1", "A2"])
        assert all(r["user_id"] == "U1" for r in df.rows)

    def test_to_csv_uses_declared_column_order(self):
        df = TaskDataframe(columns=["a", "b"])
        df.rows = [{"a": "1", "b": "2", "extra": "ignored"}]
        lines = df.to_csv().strip().splitlines()
        assert lines[0] == "a,b"
        assert lines[1] == "1,2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _example_template() -> TaskTemplate:
    """The 2-execution worked example: user_id → account_id (expand) → balance/status."""
    fields = {
        "user_id": TemplateField(label="User ID", source="user", kind=ColumnKind.SCALAR),
        "account_id": TemplateField(label="Account ID", source="generated", kind=ColumnKind.ROW, expand=True),
        "balance": TemplateField(label="Balance", source="generated", kind=ColumnKind.SCALAR),
        "status": TemplateField(label="Status", source="generated", kind=ColumnKind.SCALAR),
    }
    return TaskTemplate(
        name="Test",
        description="Pull accounts and enrich each.",
        fields=fields,
        export=["account_id", "balance", "status"],
        executions=[
            TaskExecution(
                id=1, title="Get accounts", tool="cua", system="iWeb",
                foreach="user_id", uses=["user_id"], writes=["account_id"],
                steps="Open <user_id>（User ID） account list.",
                resolved_writes={"account_id": fields["account_id"]},
            ),
            TaskExecution(
                id=2, title="Get balance", tool="cua", system="iWeb",
                foreach="account_id", uses=["account_id"], writes=["balance", "status"],
                steps="Open profile for <account_id>（Account ID）.",
                resolved_writes={"balance": fields["balance"], "status": fields["status"]},
            ),
        ],
    )


_USER_VALUES = {"user_id": "12345"}


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
# TaskRunner — happy path
# ---------------------------------------------------------------------------

class TestTaskRunnerHappyPath:
    def test_two_execution_example_builds_expected_dataframe(self, tmp_path):
        template = _example_template()
        factory = _scripted_factory([
            {"account_id": ["A001", "A002", "A003"]},
            {"balance": ["10.00"], "status": ["active"]},
            {"balance": ["20.00"], "status": ["closed"]},
            {"balance": ["30.00"], "status": ["active"]},
        ])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        events = list(runner.run_task())

        assert runner.dataframe.rows == [
            {"user_id": "12345", "account_id": "A001", "balance": "10.00", "status": "active"},
            {"user_id": "12345", "account_id": "A002", "balance": "20.00", "status": "closed"},
            {"user_id": "12345", "account_id": "A003", "balance": "30.00", "status": "active"},
        ]

        csv_path = tmp_path / "task_result.csv"
        assert csv_path.exists()
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "Account ID,Balance,Status"  # user_id excluded — not in export
        assert "A002,20.00,closed" in lines[2]

        types = [e["type"] for e in events]
        assert types.count("execution_complete") == 2
        assert types[-1] == "task_complete"
        assert events[-1]["success"] is True

    def test_seed_row_carries_user_values_into_substitution(self, tmp_path):
        template = _example_template()
        factory = _scripted_factory([
            {"account_id": ["A001"]},
            {"balance": ["1"], "status": ["x"]},
        ])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        list(runner.run_task())
        assert "12345" in factory.calls[0]["task_string"]
        assert "A001" in factory.calls[1]["task_string"]

    def test_chained_explode_user_to_accounts_to_transactions(self, tmp_path):
        """user_id → accounts (expand) → transactions (expand) → balance."""
        fields = {
            "user_id": TemplateField(label="User ID", source="user", kind=ColumnKind.SCALAR),
            "accounts": TemplateField(label="Accounts", source="generated", kind=ColumnKind.ROW, expand=True),
            "transactions": TemplateField(label="Transactions", source="generated", kind=ColumnKind.ROW, expand=True),
            "balance": TemplateField(label="Balance", source="generated", kind=ColumnKind.SCALAR),
        }
        template = TaskTemplate(
            name="", description="chain",
            fields=fields,
            export=["accounts", "transactions", "balance"],
            executions=[
                TaskExecution(
                    id=1, title="", tool="cua", system="iWeb",
                    foreach="user_id", uses=["user_id"], writes=["accounts"],
                    steps="<user_id>（User ID）",
                    resolved_writes={"accounts": fields["accounts"]},
                ),
                TaskExecution(
                    id=2, title="", tool="cua", system="iWeb",
                    foreach="accounts", uses=["accounts"], writes=["transactions"],
                    steps="<accounts>（Accounts）",
                    resolved_writes={"transactions": fields["transactions"]},
                ),
                TaskExecution(
                    id=3, title="", tool="cua", system="iWeb",
                    foreach="transactions", uses=["transactions"], writes=["balance"],
                    steps="<transactions>（Transactions）",
                    resolved_writes={"balance": fields["balance"]},
                ),
            ],
        )
        factory = _scripted_factory([
            {"accounts": ["A1", "A2"]},
            {"transactions": ["T1", "T2"]},
            {"transactions": ["T3"]},
            {"balance": ["10"]},
            {"balance": ["20"]},
            {"balance": ["30"]},
        ])
        runner = TaskRunner(template, factory, tmp_path, user_values={"user_id": "U1"})
        list(runner.run_task())

        assert len(runner.dataframe.rows) == 3
        observed = [(r["accounts"], r["transactions"], r["balance"]) for r in runner.dataframe.rows]
        assert observed == [("A1", "T1", "10"), ("A1", "T2", "20"), ("A2", "T3", "30")]

    def test_expanding_user_field_seeds_multiple_rows(self, tmp_path):
        fields = {
            "doc": TemplateField(label="Document", source="user", kind=ColumnKind.FILE, expand=True),
            "status": TemplateField(label="Status", source="generated", kind=ColumnKind.SCALAR),
        }
        template = TaskTemplate(
            name="", description="",
            fields=fields,
            export=["doc", "status"],
            executions=[
                TaskExecution(
                    id=1, title="", tool="cua", system="iWeb",
                    foreach="doc", uses=["doc"], writes=["status"], steps="",
                    resolved_writes={"status": fields["status"]},
                ),
            ],
        )
        factory = _scripted_factory([
            {"status": ["done"]},
            {"status": ["done"]},
        ])

        runner = TaskRunner(
            template, factory, tmp_path,
            user_values={"doc": ["a.pdf", "b.pdf"]},
        )
        list(runner.run_task())

        assert runner.dataframe.rows == [
            {"doc": "a.pdf", "status": "done"},
            {"doc": "b.pdf", "status": "done"},
        ]


# ---------------------------------------------------------------------------
# TaskRunner — write semantics
# ---------------------------------------------------------------------------

class TestMergeFactsSemantics:
    def _simple_template(self, fields: dict, writes: list) -> TaskTemplate:
        """One-execution template for write-semantics tests."""
        return TaskTemplate(
            name="", description="",
            fields=fields,
            export=list(fields.keys()),
            executions=[TaskExecution(
                id=1, title="", tool="cua", system="iWeb",
                foreach=list(fields.keys())[0],
                uses=[], writes=writes, steps="",
                resolved_writes={k: fields[k] for k in writes},
            )],
        )

    def test_kind_row_expand_false_joins_into_one_cell(self, tmp_path):
        """A non-expanding multi-value output collapses to a comma-joined cell."""
        fields = {"tags": TemplateField(label="Tags", source="generated", kind=ColumnKind.ROW, expand=False)}
        template = self._simple_template(fields, ["tags"])
        factory = _scripted_factory([{"tags": ["red", "green", "blue"]}])

        runner = TaskRunner(template, factory, tmp_path, user_values={})
        list(runner.run_task())
        assert runner.dataframe.rows == [{"tags": "red, green, blue"}]

    def test_scalar_with_one_value_writes_that_value(self, tmp_path):
        fields = {"name": TemplateField(label="Name", source="generated", kind=ColumnKind.SCALAR)}
        template = self._simple_template(fields, ["name"])
        factory = _scripted_factory([{"name": ["Alice"]}])

        runner = TaskRunner(template, factory, tmp_path, user_values={})
        list(runner.run_task())
        assert runner.dataframe.rows == [{"name": "Alice"}]

    def test_scalar_lands_on_seed_row_before_expand(self, tmp_path):
        """In one execution: scalar set first, then expand — new rows inherit it."""
        fields = {
            "name": TemplateField(label="Name", source="generated", kind=ColumnKind.SCALAR),
            "account": TemplateField(label="Account", source="generated", kind=ColumnKind.ROW, expand=True),
        }
        template = TaskTemplate(
            name="", description="",
            fields=fields,
            export=["name", "account"],
            executions=[TaskExecution(
                id=1, title="", tool="cua", system="iWeb",
                foreach="name", uses=[], writes=["name", "account"], steps="",
                resolved_writes={"name": fields["name"], "account": fields["account"]},
            )],
        )
        factory = _scripted_factory([{"name": ["Alice"], "account": ["A1", "A2"]}])

        runner = TaskRunner(template, factory, tmp_path, user_values={})
        list(runner.run_task())
        assert runner.dataframe.rows == [
            {"name": "Alice", "account": "A1"},
            {"name": "Alice", "account": "A2"},
        ]

    def test_unknown_output_key_logged_and_skipped(self, tmp_path, caplog):
        fields = {"known": TemplateField(label="Known", source="generated", kind=ColumnKind.SCALAR)}
        template = self._simple_template(fields, ["known"])
        factory = _scripted_factory([{"known": ["v"], "stray": ["x"]}])

        runner = TaskRunner(template, factory, tmp_path, user_values={})
        with caplog.at_level("WARNING"):
            list(runner.run_task())
        assert runner.dataframe.rows == [{"known": "v"}]
        assert any("stray" in r.message for r in caplog.records)

    def test_empty_scalar_does_not_blank_existing_cell(self, tmp_path):
        """A scalar fact with empty values is skipped, not written as \"\"."""
        fields = {"status": TemplateField(label="Status", source="generated", kind=ColumnKind.SCALAR)}
        template = TaskTemplate(
            name="", description="",
            fields=fields,
            export=["status"],
            executions=[
                TaskExecution(
                    id=1, title="", tool="cua", system="iWeb",
                    foreach="status", uses=[], writes=["status"], steps="",
                    resolved_writes={"status": fields["status"]},
                ),
                TaskExecution(
                    id=2, title="", tool="cua", system="iWeb",
                    foreach="status", uses=[], writes=["status"], steps="",
                    resolved_writes={"status": fields["status"]},
                ),
            ],
        )
        factory = _scripted_factory([
            {"status": ["active"]},
            {"status": []},  # empty list must NOT blank the prior value
        ])
        runner = TaskRunner(template, factory, tmp_path, user_values={})
        list(runner.run_task())
        assert runner.dataframe.rows == [{"status": "active"}]

    def test_undeclared_fact_is_skipped(self, tmp_path, caplog):
        """A fact key not declared in execution.writes is dropped at runtime."""
        fields = {
            "primary": TemplateField(label="Primary", source="generated", kind=ColumnKind.ROW, expand=True),
            "other": TemplateField(label="Other", source="generated", kind=ColumnKind.ROW, expand=True),
        }
        template = TaskTemplate(
            name="", description="",
            fields=fields,
            export=["primary", "other"],
            executions=[TaskExecution(
                id=1, title="", tool="cua", system="iWeb",
                foreach="primary", uses=[], writes=["primary"],  # only primary
                steps="",
                resolved_writes={"primary": fields["primary"]},
            )],
        )
        factory = _scripted_factory([
            {"primary": ["A", "B"], "other": ["C", "D", "E"]},
        ])
        runner = TaskRunner(template, factory, tmp_path, user_values={})
        with caplog.at_level("WARNING"):
            list(runner.run_task())
        assert len(runner.dataframe.rows) == 2
        assert all("other" not in r for r in runner.dataframe.rows)
        assert any("other" in r.message for r in caplog.records)

    def test_empty_expand_removes_seed_row(self, tmp_path):
        """Agent returning zero row values ⇒ downstream rows = 0; CSV header only."""
        template = _example_template()
        factory = _scripted_factory([{"account_id": []}])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        list(runner.run_task())
        assert runner.dataframe.rows == []
        # Execution 2 ran zero times because no rows
        assert [c["execution_id"] for c in factory.calls] == [1]


# ---------------------------------------------------------------------------
# TaskRunner — failure modes
# ---------------------------------------------------------------------------

class TestTaskRunnerFailures:
    def test_error_event_aborts_task(self, tmp_path):
        template = _example_template()
        factory = _scripted_factory([
            {"account_id": ["A1", "A2", "A3"]},
            {"balance": ["10"], "status": ["active"]},
            "error",  # row 1 of execution 2 fails
        ])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        events = list(runner.run_task())

        assert [c["execution_id"] for c in factory.calls] == [1, 2, 2]
        proc_complete = [e for e in events if e["type"] == "task_complete"][0]
        assert proc_complete["success"] is False
        assert proc_complete["csv_path"] is None
        assert not (tmp_path / "task_result.csv").exists()

    def test_agent_returns_without_complete_treated_as_failure(self, tmp_path):
        template = _example_template()
        factory = _scripted_factory([
            {"account_id": ["A1"]},
            "crash",
        ])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        events = list(runner.run_task())
        proc_complete = [e for e in events if e["type"] == "task_complete"][0]
        assert proc_complete["success"] is False
        assert not (tmp_path / "task_result.csv").exists()

    def test_runner_drops_complete_events_from_agent(self, tmp_path):
        template = _example_template()
        factory = _scripted_factory([
            {"account_id": ["A1"]},
            {"balance": ["1"], "status": ["x"]},
        ])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        events = list(runner.run_task())
        assert all(e["type"] != "complete" for e in events)

    def test_runner_forwards_non_complete_events(self, tmp_path):
        template = _example_template()
        factory = _scripted_factory(
            completes=[{"account_id": ["A1"]}, {"balance": ["1"], "status": ["x"]}],
            extra_events=[
                [{"type": "step", "step_num": 1}, {"type": "thinking", "response_text": "..."}],
                [{"type": "step", "step_num": 1}],
            ],
        )

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        events = list(runner.run_task())
        forwarded = [e for e in events if e["type"] in ("step", "thinking")]
        assert len(forwarded) == 3


# ---------------------------------------------------------------------------
# TaskRunner.run_once — single-agent preview entry point used by the UI
# "Test execution" affordance.
# ---------------------------------------------------------------------------


class TestRunOnce:
    def test_run_once_yields_complete_with_facts(self, tmp_path):
        """run_once surfaces the complete event including facts."""
        template = _example_template()
        factory = _scripted_factory([{"account_id": ["A001", "A002"]}])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        events = list(runner.run_once(template.executions[0]))

        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["facts"] == {"account_id": ["A001", "A002"]}

    def test_run_once_does_not_merge_facts_into_dataframe(self, tmp_path):
        """run_once is a preview — dataframe is not mutated."""
        template = _example_template()
        factory = _scripted_factory([{"account_id": ["A001"]}])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        list(runner.run_once(template.executions[0]))

        assert "account_id" not in runner.dataframe.rows[0]

    def test_run_once_does_not_write_csv(self, tmp_path):
        """run_once is a preview — no CSV written."""
        template = _example_template()
        factory = _scripted_factory([{"account_id": ["A1"]}])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        list(runner.run_once(template.executions[0]))

        assert not (tmp_path / "task_result.csv").exists()

    def test_run_once_uses_seed_row_by_default(self, tmp_path):
        """Without explicit row_idx, runs against row 0 (the seed row)."""
        template = _example_template()
        factory = _scripted_factory([{"account_id": ["A001"]}])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        list(runner.run_once(template.executions[0]))

        assert factory.calls[0]["execution_id"] == 1
        assert "12345" in factory.calls[0]["task_string"]

    def test_run_once_error_event_surfaces(self, tmp_path):
        """Error events from the agent are forwarded unchanged."""
        template = _example_template()
        factory = _scripted_factory(["error"])

        runner = TaskRunner(template, factory, tmp_path, user_values=_USER_VALUES)
        events = list(runner.run_once(template.executions[0]))

        assert any(e["type"] == "error" for e in events)
