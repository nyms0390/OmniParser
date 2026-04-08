"""
Structured checklist data model for Orchestrated and Task agent modes.

A Checklist is a ordered list of ChecklistItems, each with a status that can
be updated by the Reflect step.  The model can be constructed from:

- LLM JSON output (plan generation / task-parse responses)
- Free-form user text (numbered/bullet lists, or arbitrary prose)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Status values
PENDING = "pending"
IN_PROGRESS = "in_progress"
DONE = "done"
SKIPPED = "skipped"

_STATUS_SYMBOLS: Dict[str, str] = {
    DONE: "✅",
    IN_PROGRESS: "🔄",
    SKIPPED: "⏭️",
    PENDING: "⬜",
}


@dataclass
class ChecklistItem:
    """A single step in a checklist.

    Attributes:
        id: 1-based integer identifier.
        step: Human-readable description of the step.
        status: Current status — one of ``pending``, ``in_progress``,
            ``done``, ``skipped``.
        verification_hint: Optional hint for how to verify the step is done
            (populated by the plan LLM; used by the Reflect step).
    """

    id: int
    step: str
    status: str = PENDING
    verification_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "status": self.status,
            "verification_hint": self.verification_hint,
        }


@dataclass
class Checklist:
    """Ordered collection of :class:`ChecklistItem` objects.

    Usage::

        checklist = Checklist.from_llm_json(llm_response)
        checklist.apply_updates([{"id": 1, "status": "done"}])
        if checklist.all_done():
            ...
    """

    items: List[ChecklistItem] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_done(self) -> bool:
        """Return *True* when every item is ``done`` or ``skipped``."""
        return bool(self.items) and all(
            item.status in (DONE, SKIPPED) for item in self.items
        )

    def get_active(self) -> Optional[ChecklistItem]:
        """Return the first item that is not yet ``done`` or ``skipped``."""
        for item in self.items:
            if item.status not in (DONE, SKIPPED):
                return item
        return None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def apply_updates(self, updates: List[dict]) -> None:
        """Apply status updates from a Reflect response.

        Args:
            updates: List of ``{"id": int, "status": str}`` dicts.  Unknown
                ids and unknown fields are silently ignored.
        """
        id_map = {item.id: item for item in self.items}
        for upd in updates:
            item = id_map.get(upd.get("id"))
            if item is None:
                continue
            if "status" in upd:
                item.status = upd["status"]
            if "verification_hint" in upd:
                item.verification_hint = upd["verification_hint"]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> list:
        """Serialise to a list of dicts (JSON-serialisable)."""
        return [item.to_dict() for item in self.items]

    def to_prompt_text(self, collapse_completed: bool = False) -> str:
        """Format the checklist as a human-readable progress block.

        Args:
            collapse_completed: When *True*, done and skipped items are
                rendered as a single line without the ``verify when done``
                hint.  Use this for the REFLECT prompt to reduce noise while
                keeping full detail for pending and in-progress steps.
        """
        lines = ["Checklist:"]
        for item in self.items:
            sym = _STATUS_SYMBOLS.get(item.status, "⬜")
            lines.append(f"  {sym} [{item.id}] {item.step}")
            if item.verification_hint and not (
                collapse_completed and item.status in (DONE, SKIPPED)
            ):
                lines.append(f"       verify when done: {item.verification_hint}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_llm_json(cls, raw_json: str) -> "Checklist":
        """Parse the LLM's JSON plan/task-parse response.

        Primary format (``response_format``-enforced calls)::

            {
              "steps": [
                {"id": 1, "step": "...", "verification_hint": "..."},
                ...
              ]
            }

        Also accepts a bare array for backwards compatibility::

            [{"id": 1, "step": "...", "verification_hint": "..."}, ...]

        Falls back to a single-item checklist containing the raw text on
        any parse failure.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", raw_json).strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                data = data.get("steps", [])
            if isinstance(data, list):
                items = []
                for i, entry in enumerate(data):
                    if isinstance(entry, dict) and entry.get("step"):
                        items.append(
                            ChecklistItem(
                                id=int(entry.get("id", i + 1)),
                                step=str(entry["step"]),
                                verification_hint=str(
                                    entry.get("verification_hint", "")
                                ),
                            )
                        )
                if items:
                    return cls(items=items)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # Fallback: single item
        return cls(items=[ChecklistItem(id=1, step=raw_json.strip())])

    @classmethod
    def from_user_text(cls, text: str) -> "Checklist":
        """Parse free-form user text into a checklist.

        Supports:

        * Numbered lists — ``1. step`` or ``1) step``
        * Bullet lists   — ``- step``, ``* step``, ``• step``
        * Falls back to a single-item checklist for unstructured text.

        A step may be followed by an indented ``verify:`` continuation line
        that supplies a :attr:`~ChecklistItem.verification_hint`::

            1. Click the Submit button
               verify: "Success" toast appears at the top of the page

        The caller can detect the fallback case by checking
        ``len(checklist.items) == 1``.
        """
        raw_lines = text.strip().splitlines()
        verify_re = re.compile(r"^\s+verify:\s+(.+)$", re.IGNORECASE)

        def _attach_hints(
            items: List[ChecklistItem],
            lines: List[str],
            step_re: re.Pattern,
        ) -> None:
            """Walk raw lines; attach indented verify: lines to the preceding step."""
            item_idx = -1
            for ln in lines:
                if step_re.match(ln.strip()):
                    item_idx += 1
                else:
                    m = verify_re.match(ln)
                    if m and 0 <= item_idx < len(items):
                        items[item_idx].verification_hint = m.group(1).strip()

        # Numbered list
        numbered_re = re.compile(r"^\d+[.)]\s+(.+)$")
        items = [
            ChecklistItem(id=idx + 1, step=m.group(1))
            for idx, m in enumerate(
                numbered_re.match(ln.strip())
                for ln in raw_lines
                if numbered_re.match(ln.strip())
            )
        ]
        if items:
            _attach_hints(items, raw_lines, numbered_re)
            return cls(items=items)

        # Bullet list
        bullet_re = re.compile(r"^[-*•]\s+(.+)$")
        items = [
            ChecklistItem(id=idx + 1, step=m.group(1))
            for idx, m in enumerate(
                bullet_re.match(ln.strip())
                for ln in raw_lines
                if bullet_re.match(ln.strip())
            )
        ]
        if items:
            _attach_hints(items, raw_lines, bullet_re)
            return cls(items=items)

        # Single-item fallback
        return cls(items=[ChecklistItem(id=1, step=text.strip())])
