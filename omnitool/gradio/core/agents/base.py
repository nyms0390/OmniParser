"""
BaseAgent — abstract base class for ReActAgent.

Provides shared infrastructure (screen capture, tool execution, cost tracking,
field extraction helpers) without prescribing a loop structure.

Subclasses implement ``run()`` with their own loop:

    ReActAgent — native tool-calling loop with pluggable GroundingStrategy
"""

import base64
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from PIL import Image

from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.core.agents.preprocessing import PreprocessingMode, preprocess_b64
from omnitool.gradio.config import (
    AgentMode,
    SCREENSHOT_MAX_WIDTH,
    TaskProcedure,
    get_llm_config,
    get_pricing,
)
from omnitool.gradio.core.agents.image_utils import _crop_b64
from omnitool.gradio.services.state import AppState

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemory:
    """Transient agent state for a single run.

    Attributes:
        task: The user's task string.
        facts: Key-value pairs committed to memory via ``save_field``.
        staged_reads: Verified values from ``read_field``, keyed by field_name,
            pending commit via ``save_field``.
        trajectory: Ordered list of step data dicts (action history).
        parsed_screen: Most recent captured screen state.
    """

    task: Optional[str] = None
    facts: Dict[str, List[str]] = field(default_factory=dict)
    staged_reads: Dict[str, List[str]] = field(default_factory=dict)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    parsed_screen: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Abstract base class for OmniParser agents.

    Provides shared infrastructure and helpers. Subclasses implement
    ``run()`` with their own loop.
    """

    def __init__(
        self,
        model_name: str,
        llm_client: BaseLLMClient,
        state: AppState,
        tools_collection,
        save_folder: Path,
        mode: AgentMode = AgentMode.INTERACTIVE,
        platform: str = "windows",
        max_steps: int = 20,
        action_delay: float = 1.5,
        gta1_client: Optional[GTA1Client] = None,
        provider: Optional[str] = None,
        preprocessing_mode: PreprocessingMode = PreprocessingMode.RAW,
        task_procedure: Optional[TaskProcedure] = None,
    ):
        self.model_name = model_name
        self.provider = provider or ""
        self.llm_client = llm_client
        self.state = state
        self.tools_collection = tools_collection
        self.save_folder = Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.platform = platform
        self.max_steps = max_steps
        self.action_delay = action_delay
        self.gta1_client = gta1_client
        self.screenshot_max_width = SCREENSHOT_MAX_WIDTH
        self.preprocessing_mode = preprocessing_mode

        # LLM config for cost calculation
        try:
            self.llm_config = get_llm_config(model_name)
        except ValueError:
            self.llm_config = {}

        # Usage tracking
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._focus_crop_count: int = 0
        self._start_time: Optional[datetime] = None
        self._flags: List[Dict[str, Any]] = []
        self._compact_pending: bool = False

        # Working memory
        self.working_memory = WorkingMemory()

        # Task procedure — provided at construction when running in TASK mode.
        self.task_procedure: Optional[TaskProcedure] = task_procedure

    # ------------------------------------------------------------------
    # Lifecycle & accounting
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all per-run counters and state."""
        self.step_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._focus_crop_count = 0
        self._start_time = None
        self._flags = []
        self._compact_pending = False
        self.working_memory = WorkingMemory()

    def _record_start(self) -> None:
        """Record run start time. Called once at the top of each subclass ``run()``."""
        self._start_time = datetime.now()
        self._flags = []

    def update_step_count(self):
        """Increment the step counter by one."""
        self.step_count += 1
        self._focus_crop_count = 0

    def update_token_usage(self, tokens: int):
        """Add *tokens* to the cumulative token count."""
        self.total_tokens += tokens

    def update_cost(self, cost: float):
        """Add *cost* USD to the cumulative cost total."""
        self.total_cost += cost

    def _calculate_cost(self, metadata: Dict[str, Any]) -> float:
        """Calculate cost in USD from LLM response metadata.

        Args:
            metadata: Dict returned by ``llm_client.generate()`` containing
                ``"input_tokens"`` and ``"output_tokens"`` keys.

        Returns:
            Estimated cost in USD, or ``0.0`` when pricing is unavailable.
        """
        if not self.llm_config:
            return 0.0
        rates = get_pricing(self.model_name, self.provider)
        try:
            input_tokens = metadata.get("input_tokens", 0)
            output_tokens = metadata.get("output_tokens", 0)
            input_cost = (input_tokens * rates.get("input", 0.0)) / 1_000_000
            output_cost = (output_tokens * rates.get("output", 0.0)) / 1_000_000
            return input_cost + output_cost
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def _capture_screen(self) -> Dict[str, Any]:
        """Capture and resize the current screen.

        Returns:
            Dict with keys:
            - ``raw_image_base64``: original full-resolution screenshot (base64)
            - ``resized_image_base64``: after resize, before preprocessing (base64)
            - ``preprocessed_image_base64``: after resize + preprocessing (base64)
            - ``screen_width``, ``screen_height``: actual screen dimensions
            - ``resized_screen_width``, ``resized_screen_height``: VLM image dims
        """
        computer_tool = self.tools_collection.get_tool("computer")
        if not computer_tool:
            raise ValueError("ComputerTool not available")

        screenshot_result = computer_tool.run("screenshot")
        if screenshot_result.error:
            raise ValueError(f"Screenshot failed: {screenshot_result.error}")

        screenshot_b64 = screenshot_result.base64_image
        if not screenshot_b64:
            raise ValueError("No screenshot data from ComputerTool")

        # Decode once: read dimensions and resize in a single pass.
        screen_width, screen_height = 1920, 1080
        resized_b64 = screenshot_b64
        resized_w, resized_h = screen_width, screen_height
        try:
            orig_img = Image.open(BytesIO(base64.b64decode(screenshot_b64)))
            screen_width, screen_height = orig_img.size
            if orig_img.width > self.screenshot_max_width:
                ratio = self.screenshot_max_width / orig_img.width
                new_w = self.screenshot_max_width
                new_h = int(orig_img.height * ratio)
                resized_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                buf = BytesIO()
                resized_img.save(buf, format="PNG")
                resized_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                resized_w, resized_h = new_w, new_h
                logger.debug(
                    "Screenshot resized %dx%d -> %dx%d",
                    screen_width, screen_height, new_w, new_h,
                )
            else:
                resized_w, resized_h = screen_width, screen_height
        except Exception as exc:
            logger.warning("Could not process screenshot: %s", exc)

        preprocessed_b64 = preprocess_b64(resized_b64, self.preprocessing_mode)

        return {
            "raw_image_base64":          screenshot_b64,
            "resized_image_base64":      resized_b64,
            "preprocessed_image_base64": preprocessed_b64,
            "screen_width":              screen_width,
            "screen_height":             screen_height,
            "resized_screen_width":      resized_w,
            "resized_screen_height":     resized_h,
        }

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the fully-rendered system prompt for this agent variant."""

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using the agent's tools collection.

        Args:
            tool_calls: List of tool-call dicts with at minimum a ``"tool"`` key.

        Returns:
            List of result dicts, each with ``"tool"``, ``"status"``, and either
            ``"result"`` or ``"error"``.
        """
        results: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            action = tool_call.get("action", "")
            label = f"{tool_name}.{action}" if action else tool_name
            tool = self.tools_collection.get_tool(tool_name)
            if tool is None:
                logger.info("ACT [%s] FAILED — tool not found", label)
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "error": f"Tool not found: {tool_name}",
                })
                continue
            try:
                tool_kwargs = {
                    k: v for k, v in tool_call.items()
                    if k not in ("tool", "action")
                }
                logger.info(
                    "ACT [%s] — input: %s",
                    label,
                    {k: v for k, v in tool_kwargs.items() if k != "image"},
                )
                result = tool.run(tool_call.get("action"), **tool_kwargs)
                if hasattr(result, "error") and result.error:
                    logger.info("ACT [%s] FAILED — %s", label, result.error)
                else:
                    output_preview = str(getattr(result, "output", result) or "")
                    logger.info(
                        "ACT [%s] OK — output: %s",
                        label,
                        output_preview[:200] if output_preview else "(no output)",
                    )
                results.append({"tool": tool_name, "status": "success", "result": result})
            except Exception as exc:
                logger.error("Tool execution failed for %s: %s", tool_name, exc)
                logger.info("ACT [%s] FAILED — %s", label, exc)
                results.append({"tool": tool_name, "status": "error", "error": str(exc)})
        return results

    # ------------------------------------------------------------------
    # Field extraction helpers
    # ------------------------------------------------------------------

    def _scale_to_screen(
        self,
        x: float,
        y: float,
        parsed_screen: Dict[str, Any],
    ) -> tuple:
        """Scale GTA1 coords (resized-image space) to actual screen space."""
        resized_w = parsed_screen.get("resized_screen_width") or parsed_screen.get("screen_width", 1)
        resized_h = parsed_screen.get("resized_screen_height") or parsed_screen.get("screen_height", 1)
        screen_w = parsed_screen.get("screen_width", resized_w)
        screen_h = parsed_screen.get("screen_height", resized_h)
        sx = round(x * screen_w / resized_w) if resized_w else round(x)
        sy = round(y * screen_h / resized_h) if resized_h else round(y)
        return sx, sy

    def _read_field_via_clipboard(
        self,
        field_name: str,
        description: str,
        parsed_screen: Dict[str, Any],
    ) -> str:
        """Extract a single field value using GTA1 + triple-click + clipboard.

        Uses the GTA1 grounding model to locate the field's center point in
        the resized VLM image, scales coordinates back to actual screen space,
        triple-clicks to select the content, copies with Ctrl+C, and reads
        from the clipboard.

        Returns:
            Stripped clipboard text, ``"null"`` when the field is not found or
            empty, or ``"extraction failed"`` on unexpected errors.
        """
        resized_b64 = parsed_screen.get("resized_image_base64", "")
        if not resized_b64:
            logger.warning(
                "_read_field_via_clipboard: no screenshot for %r", field_name
            )
            return "extraction failed"

        try:
            result = self.gta1_client.ground(resized_b64, description)
            rx, ry = result["x"], result["y"]
        except Exception as exc:
            logger.warning(
                "_read_field_via_clipboard: GTA1 grounding failed for %r: %s",
                field_name, exc,
            )
            return "extraction failed"

        if rx == 0 and ry == 0:
            logger.debug(
                "_read_field_via_clipboard: zero coordinates for %r — field not visible",
                field_name,
            )
            return "null"

        x, y = self._scale_to_screen(rx, ry, parsed_screen)

        logger.debug(
            "_read_field_via_clipboard: %r grounded at resized (%s, %s) -> screen (%s, %s)",
            field_name, rx, ry, x, y,
        )
        try:
            self.tools_collection.run("computer", "triple_click", coordinate=(x, y))
            self.tools_collection.run("computer", "key", text="ctrl+c")
            clip_result = self.tools_collection.run("computer", "read_clipboard")
            if clip_result.error:
                logger.warning(
                    "_read_field_via_clipboard: clipboard read error for %r: %s",
                    field_name, clip_result.error,
                )
                return "extraction failed"
            value = (clip_result.output or "").strip()
            return value if value else "null"
        except Exception as exc:
            logger.warning(
                "_read_field_via_clipboard: error extracting %r: %s", field_name, exc
            )
            return "extraction failed"

    def _correct_field_via_clipboard(
        self,
        field_name: str,
        ocr_value: List[str],
        parsed_screen: Dict[str, Any],
    ) -> List[str]:
        """Correct LLM-extracted field values using GTA1 + triple-click + clipboard.

        Uses each LLM-extracted value as the GTA1 grounding instruction to locate
        the exact element on screen, then reads the true value from the clipboard.
        Falls back to the original item string if GTA1 fails for that item.
        """
        corrected = []
        for idx, item in enumerate(ocr_value):
            item_str = str(item)
            instruction = f'the element showing "{item_str}"'
            result = self._read_field_via_clipboard(
                f"{field_name}[{idx}]", instruction, parsed_screen
            )
            corrected.append(
                result if result not in ("extraction failed", "null") else item_str
            )
        changed = sum(1 for a, b in zip(corrected, ocr_value) if str(a) != str(b))
        logger.info(
            "_correct_field_via_clipboard: %r corrected %d/%d items",
            field_name, changed, len(ocr_value),
        )
        return corrected

    # ------------------------------------------------------------------
    # read_field tool handler
    # ------------------------------------------------------------------

    def _set_fact(self, key: str, value: str | list[str], *, overwrite: bool = False) -> None:
        """Write a value into working_memory.facts.

        - overwrite=False: appends to the list; *value* must be a plain ``str``.
        - overwrite=True: replaces the stored list entirely. When *value* is already
          a ``list[str]`` (e.g. from ``AggregateOperation.NONE``) it is stored
          as-is; a plain ``str`` is wrapped in a single-element list.
        """
        if overwrite:
            stored = value if isinstance(value, list) else [value]
            existing = self.working_memory.facts.get(key)
            if existing is not None and existing != stored:
                logger.warning("FACTS — overwriting %r: %r → %r", key, existing, stored)
            self.working_memory.facts[key] = stored
        else:
            if not isinstance(value, str):
                raise TypeError(
                    f"_set_fact append path requires str, got {type(value).__name__!r}"
                )
            self.working_memory.facts.setdefault(key, []).append(value)

    def _handle_read_field(self, tc_args: Dict[str, Any]) -> Tuple[str, Dict[str, List[str]]]:
        """Read and optionally verify one or more screen values via clipboard correction.

        Each item is appended to ``staged_reads[field_name]`` — multiple items with
        the same ``field_name`` accumulate in order across calls and within a single
        call. Call ``_handle_save_field`` to drain the staged list into
        ``working_memory.facts``.
        """
        items = tc_args.get("fields", [])
        if not items:
            return "Error: fields list is required and must not be empty.", {}

        results = []
        read_values: Dict[str, List[str]] = {}

        for item in items:
            field_name = item.get("field_name", "")
            value = item.get("value", "")
            target = item.get("target")

            if not field_name:
                results.append("Error: field_name is required.")
                continue

            out = self.task_procedure.get_output(field_name) if self.task_procedure else None
            if self.task_procedure is not None and out is None:
                results.append(
                    f"Error: field_name '{field_name}' is not declared in the task procedure."
                )
                continue
            use_correction = out.clipboard_correction if out else True

            corrected_list = [value]
            if value and use_correction and self.gta1_client and target:
                try:
                    corrected_list = self._correct_field_via_clipboard(
                        field_name, [value], self.working_memory.parsed_screen or {}
                    )
                except Exception as exc:
                    logger.warning("Field correction failed for '%s': %s", field_name, exc)

            corrected = corrected_list[0] if corrected_list else value

            self.working_memory.staged_reads.setdefault(field_name, []).append(corrected)

            n = len(self.working_memory.staged_reads[field_name])
            if n > 1:
                logger.info(
                    "STAGED_READS — appended to '%s' (now %d values): %r",
                    field_name, n, corrected,
                )
            else:
                logger.info("STAGED_READS — first value for '%s': %r", field_name, corrected)

            results.append(f"Read: {field_name} = {corrected}")
            read_values.setdefault(field_name, []).append(corrected)

        return "\n".join(results), read_values

    def _handle_save_field(self, tc_args: Dict[str, Any]) -> Tuple[str, Dict[str, List[str]]]:
        """Commit all staged values for each requested field to ``working_memory.facts``.

        Drains the entire ``staged_reads[field_name]`` list: dynamic fields append
        each value via ``_set_fact(..., overwrite=False)``; scalar fields keep only
        the last staged value via ``_set_fact(..., overwrite=True)``. Returns a
        tool-result error string if a ``field_name`` has no staged reads.
        """
        items = tc_args.get("fields", [])
        if not items:
            return "Error: fields list is required and must not be empty.", {}

        results = []
        captured: Dict[str, List[str]] = {}

        for item in items:
            field_name = item.get("field_name", "")
            if not field_name:
                results.append("Error: field_name is required.")
                continue

            # Pop the entire accumulated list
            staged_values = self.working_memory.staged_reads.pop(field_name, [])
            if not staged_values:
                results.append(
                    f"Error: no staged value for '{field_name}' — call read_field first."
                )
                continue

            logger.info("SAVE_FIELD '%s': committing %d value(s)", field_name, len(staged_values))

            out = self.task_procedure.get_output(field_name) if self.task_procedure else None
            if self.task_procedure is not None and out is None:
                results.append(
                    f"Error: field_name '{field_name}' is not declared in the task procedure."
                )
                continue
            is_dynamic = out.is_dynamic if out else False

            captured[field_name] = staged_values

            if is_dynamic:
                for value in staged_values:
                    self._set_fact(field_name, value, overwrite=False)
                results.append(f"Saved {len(staged_values)} values to dynamic field '{field_name}'")
            else:
                if len(staged_values) > 1:
                    logger.warning(
                        "SAVE_FIELD '%s': scalar field has %d staged values — "
                        "using last: %r (dropped: %r)",
                        field_name, len(staged_values), staged_values[-1], staged_values[:-1],
                    )
                value = staged_values[-1]
                self._set_fact(field_name, value, overwrite=True)
                results.append(f"Saved: {field_name} = {value}")

        return "\n".join(results), captured

    def _apply_template_aggregates(self) -> None:
        """Compute template-declared aggregates and write results into facts.

        Called once at finish. For each output in ``task_procedure`` that has
        an ``aggregate`` config, reads the accumulated list from
        ``working_memory.facts[aggregate.source]``, applies the operation over
        numeric values, and stores the result under ``output.key``.
        """
        if self.task_procedure is None:
            return
        for out in self.task_procedure.outputs:
            if out.aggregate is None:
                continue
            source_values = self.working_memory.facts.get(out.aggregate.source, [])
            if not all(isinstance(v, str) for v in source_values):
                raise TypeError(
                    f"facts invariant violated for {out.key!r}: "
                    f"non-str values in {source_values!r}"
                )
            if not source_values:
                logger.warning(
                    "Aggregate for %r: no values in source %r", out.key, out.aggregate.source
                )
                continue
            try:
                result = out.aggregate.operation.apply(source_values)
            except (NotImplementedError, ValueError) as exc:
                logger.error(
                    "Aggregate operation %r failed for output %r: %s",
                    out.aggregate.operation, out.key, exc,
                )
                continue
            self._set_fact(out.key, result, overwrite=True)
            logger.info(
                "Aggregate %r = %r (from %d values in %r)",
                out.key, result, len(source_values), out.aggregate.source,
            )

    # ------------------------------------------------------------------
    # Auxiliary tool handlers
    # ------------------------------------------------------------------

    def _handle_focus_region(self, tc_args: Dict[str, Any]) -> Optional[str]:
        """Crop the current screenshot to the requested bbox.

        Args:
            tc_args: Tool call arguments dict; expects ``bbox: [x1, y1, x2, y2]``
                in resized image pixel coordinates.

        Returns:
            Base64-encoded cropped PNG, or ``None`` when the crop cannot be
            performed (missing screenshot, invalid bbox, or crop failure).
        """
        bbox = tc_args.get("bbox", [])
        parsed = self.working_memory.parsed_screen or {}
        resized_b64 = parsed.get("resized_image_base64", "")
        if not resized_b64 or len(bbox) != 4:
            logger.warning(
                "_handle_focus_region: missing screenshot or invalid bbox %r", bbox
            )
            return None
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            crop_b64 = _crop_b64(resized_b64, x1, y1, x2, y2)
        except Exception as exc:
            logger.warning("_handle_focus_region crop failed: %s", exc)
            return None
        crop_file = f"step_{self.step_count:03d}_focus_{self._focus_crop_count:02d}.png"
        self._focus_crop_count += 1
        try:
            (self.save_folder / crop_file).write_bytes(base64.b64decode(crop_b64))
        except Exception as exc:
            logger.warning("Failed to save focus crop %s: %s", crop_file, exc)
        return crop_b64

    def _handle_mark_screenshot(self, reason: str = "") -> str:
        """Flag the current step's screenshot as important in trajectory.json.

        Returns:
            Confirmation string for the tool result message.
        """
        screenshot_file = f"step_{self.step_count:03d}.png"
        flag_record = {
            "type": "flag",
            "step": self.step_count,
            "reason": reason,
            "screenshot_file": screenshot_file,
        }
        self._flags.append(flag_record)
        trajectory_file = self.save_folder / "trajectory.json"
        try:
            with open(trajectory_file, "a", encoding="utf-8") as f:
                json.dump(flag_record, f, ensure_ascii=False)
                f.write("\n")
        except Exception as exc:
            logger.warning("Failed to save screenshot flag: %s", exc)
            return f"Error flagging {screenshot_file}: {exc}"
        return f"Flagged {screenshot_file}: {reason}"

    def _write_run_summary(self, success: bool, message: str) -> None:
        """Write summary.json to save_folder at the end of every run."""
        end_time = datetime.now()
        if self._start_time is None:
            logger.warning("_write_run_summary called before _record_start(); duration will be 0.")
        start_time = self._start_time or end_time
        total_seconds = int((end_time - start_time).total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        summary = {
            "task": self.working_memory.task or "",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "success": success,
            "message": message,
            "total_steps": self.step_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "flags": self._flags,
            "facts": dict(self.working_memory.facts),
        }
        try:
            (self.save_folder / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write summary.json: %s", exc)
        # Prefer the last-known resized screenshot to avoid a fresh (potentially
        # failing) capture — especially important in the crash handler path.
        last_b64 = None
        if self.working_memory.parsed_screen:
            last_b64 = self.working_memory.parsed_screen.get("resized_image_base64")
        if not last_b64:
            try:
                captured = self._capture_screen()
                last_b64 = captured.get("resized_image_base64") or captured.get("raw_image_base64")
            except Exception as exc:
                logger.warning("Failed to capture final screenshot: %s", exc)
        if last_b64:
            try:
                (self.save_folder / "final_screenshot.png").write_bytes(
                    base64.b64decode(last_b64)
                )
            except Exception as exc:
                logger.warning("Failed to save final screenshot: %s", exc)

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def _save_trajectory_step(self, plan_response: Dict[str, Any]):
        """Append the current step data to the in-memory trajectory and disk log.

        Writes one JSON object per line to ``<save_folder>/trajectory.json``.

        Args:
            plan_response: Dict with keys ``"response_text"``, ``"tool_calls"``,
                ``"metadata"``, and ``"cost"`` from the plan LLM call.
        """
        parsed = self.working_memory.parsed_screen or {}
        screenshot_file = f"step_{self.step_count:03d}.png"
        step_data = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "screenshot_file": screenshot_file,
            "screen_info": str(parsed.get("parsed_content_list", [])),
            "agent_response": plan_response.get("response_text", ""),
            "tool_calls": plan_response.get("tool_calls", []),
            "tokens": plan_response.get("metadata", {}).get("tokens"),
            "cost": plan_response.get("cost"),
        }
        self.working_memory.trajectory.append(step_data)
        trajectory_file = self.save_folder / "trajectory.json"
        try:
            with open(trajectory_file, "a", encoding="utf-8") as f:
                json.dump(step_data, f, ensure_ascii=False)
                f.write("\n")
        except Exception as exc:
            logger.warning("Failed to save trajectory: %s", exc)
        # Save resized screenshot alongside the trajectory entry
        resized_b64 = parsed.get("resized_image_base64", "")
        if resized_b64:
            try:
                (self.save_folder / screenshot_file).write_bytes(base64.b64decode(resized_b64))
            except Exception as exc:
                logger.warning("Failed to save screenshot %s: %s", screenshot_file, exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> Generator[Dict[str, Any], None, None]:
        """Main agentic loop. Subclasses implement the full loop."""
