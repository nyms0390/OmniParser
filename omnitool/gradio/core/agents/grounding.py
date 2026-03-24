"""
Pluggable grounding strategies for ReActAgent.

A GroundingStrategy encapsulates three responsibilities:
1. Preprocessing — raw screenshot → ScreenData (SOM annotation or pass-through).
2. Tool exposure — which OpenAI schemas to give the LLM.
3. Resolution — tool call (name + args) → execute_tool_calls dispatch dict.

Available strategies:
  OmniParserGrounding — SOM preprocessing; LLM references elements by integer box_id.
  GTA1Grounding       — raw screenshot pass-through; LLM uses natural-language target.
"""

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.core.tools.schemas import (
    GTA1_COMPUTER_TOOLS,
    OMNIPARSER_COMPUTER_TOOLS,
    POSITIONAL_ACTIONS,
)

# Crosshair appearance (mirrors GTA1/scripts/demo.py and GTAAgent)
_CROSSHAIR_RADIUS = 18
_CROSSHAIR_COLOR  = (255, 50, 50)
_CROSSHAIR_WIDTH  = 3
_DOT_RADIUS       = 5

logger = logging.getLogger(__name__)


@dataclass
class ScreenData:
    """Normalised screen observation returned by :meth:`GroundingStrategy.preprocess`.

    Attributes:
        raw_image_b64:     Resized raw screenshot (base64 PNG).
        display_image_b64: Image shown to the LLM — SOM-annotated for OmniParser,
                           same as raw for GTA1.
        elements:          Parsed element list from OmniParser; empty for GTA1.
        screen_width:      Original (unresized) screen width in pixels.
        screen_height:     Original (unresized) screen height in pixels.
        resized_width:     Width of the image sent to the LLM.
        resized_height:    Height of the image sent to the LLM.
    """

    raw_image_b64: str
    display_image_b64: str
    elements: List[Dict[str, Any]] = field(default_factory=list)
    screen_width: int = 1920
    screen_height: int = 1080
    resized_width: int = 1920
    resized_height: int = 1080


class GroundingStrategy(ABC):
    """Abstract base for all grounding strategies."""

    @abstractmethod
    def preprocess(
        self,
        raw_b64: str,
        screen_width: int,
        screen_height: int,
        resized_width: int,
        resized_height: int,
    ) -> ScreenData:
        """Preprocess a raw screenshot into a :class:`ScreenData`.

        Args:
            raw_b64:        Resized raw screenshot (base64 PNG).
            screen_width:   Original screen width in pixels.
            screen_height:  Original screen height in pixels.
            resized_width:  Width of the VLM-facing image.
            resized_height: Height of the VLM-facing image.
        """

    @abstractmethod
    def get_tools(self) -> List[dict]:
        """Return the OpenAI tool schemas for computer actions (no finish tool)."""

    def resolve(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        screen_data: ScreenData,
    ) -> Dict[str, Any]:
        """Resolve a tool call to an execute_tool_calls dispatch dict.

        Handles non-positional tools (type_text, key_press, scroll, wait) directly.
        Delegates positional tools to :meth:`_resolve_positional`.

        Returns:
            Dict with ``"tool"``, ``"action"``, and action-specific kwargs
            (``"coordinate"``, ``"text"``, ``"key"``, ``"direction"``) understood
            by :meth:`BaseAgent.execute_tool_calls`.

        Raises:
            ValueError: If required arguments are missing or resolution fails.
        """
        dispatch: Dict[str, Any] = {"tool": "computer", "action": tool_name}

        if tool_name in POSITIONAL_ACTIONS:
            dispatch["coordinate"] = self._resolve_positional(tool_name, arguments, screen_data)
        elif tool_name == "type_text":
            dispatch["action"] = "type"
            dispatch["text"] = arguments.get("text", "")
        elif tool_name == "key_press":
            dispatch["action"] = "key"
            dispatch["text"] = arguments.get("key", "")
        elif tool_name == "scroll":
            direction = arguments.get("direction", "down")
            dispatch["action"] = f"scroll_{direction}"
            dispatch["amount"] = int(arguments.get("amount", 1))
        # "wait" — bare dispatch, no extra args needed

        return dispatch

    @abstractmethod
    def _resolve_positional(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        screen_data: ScreenData,
    ) -> List[int]:
        """Resolve a positional action to ``[x, y]`` screen coordinates.

        Args:
            tool_name:   Name of the positional tool (e.g. ``"left_click"``).
            arguments:   Parsed JSON arguments from the tool call.
            screen_data: Current :class:`ScreenData` from :meth:`preprocess`.

        Raises:
            ValueError: If required arguments are missing or resolution fails.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logging and UI (e.g. ``"omniparser"``)."""

    @property
    @abstractmethod
    def element_reference_hint(self) -> str:
        """One-line description of how the LLM should reference screen elements."""

    @property
    def last_grounding_events(self) -> List[Dict[str, Any]]:
        """Grounding events from the most recent :meth:`resolve` call.

        Non-empty only for strategies that perform visual grounding (e.g.
        :class:`GTA1Grounding`).  Used by :class:`ReActAgent` to emit
        ``grounding`` events so the UI can show the crosshair-annotated image.
        """
        return []


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _draw_crosshair(image_b64: str, x: int, y: int) -> str:
    """Draw a red crosshair + dot at *(x, y)* on a base64-encoded PNG.

    Coordinates are in the image's own pixel space (i.e. the resized screenshot).
    Returns an empty string on failure so callers can treat it as optional.
    """
    try:
        img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGBA")
        draw = ImageDraw.Draw(img)
        r, w = _CROSSHAIR_RADIUS, _CROSSHAIR_WIDTH
        draw.line([(x - r, y), (x + r, y)], fill=_CROSSHAIR_COLOR, width=w)
        draw.line([(x, y - r), (x, y + r)], fill=_CROSSHAIR_COLOR, width=w)
        dr = _DOT_RADIUS
        draw.ellipse([(x - dr, y - dr), (x + dr, y + dr)], fill=_CROSSHAIR_COLOR)
        out = BytesIO()
        img.convert("RGB").save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode()
    except Exception as exc:
        logger.warning("_draw_crosshair failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# OmniParser grounding
# ---------------------------------------------------------------------------

class OmniParserGrounding(GroundingStrategy):
    """SOM-based grounding via OmniParserClient.

    Preprocesses every screenshot through OmniParser to produce a
    set-of-marks annotated image and a structured element list.
    The LLM identifies elements by integer ``box_id`` (0-based index).
    """

    def __init__(self, omniparser_client: OmniParserClient) -> None:
        self._client = omniparser_client

    @property
    def name(self) -> str:
        return "omniparser"

    @property
    def element_reference_hint(self) -> str:
        return (
            "The annotated screenshot labels each UI element with a numeric box ID. "
            "The element list shows each element as: `<id>: <type>, <description>`. "
            "Use the `box_id` parameter (integer) to reference the element you want."
        )

    def preprocess(
        self,
        raw_b64: str,
        screen_width: int,
        screen_height: int,
        resized_width: int,
        resized_height: int,
    ) -> ScreenData:
        try:
            result = self._client.parse_screenshot(raw_b64)
            som_b64 = result.get("labeled_screenshot_base64", "") or raw_b64
            elements = result.get("parsed_content_list", [])
        except Exception as exc:
            logger.warning("OmniParser preprocessing failed: %s — using raw screenshot", exc)
            som_b64 = raw_b64
            elements = []

        return ScreenData(
            raw_image_b64=raw_b64,
            display_image_b64=som_b64,
            elements=elements,
            screen_width=screen_width,
            screen_height=screen_height,
            resized_width=resized_width,
            resized_height=resized_height,
        )

    def get_tools(self) -> List[dict]:
        return OMNIPARSER_COMPUTER_TOOLS

    def _resolve_positional(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        screen_data: ScreenData,
    ) -> List[int]:
        box_id = arguments.get("box_id")
        if box_id is None:
            raise ValueError(f"'{tool_name}' requires box_id")
        try:
            bbox = screen_data.elements[int(box_id)]["bbox"]
            cx = int((bbox[0] + bbox[2]) / 2 * screen_data.screen_width)
            cy = int((bbox[1] + bbox[3]) / 2 * screen_data.screen_height)
            return [cx, cy]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"box_id {box_id} resolution failed: {exc}") from exc


# ---------------------------------------------------------------------------
# GTA1 grounding
# ---------------------------------------------------------------------------

class GTA1Grounding(GroundingStrategy):
    """Natural-language grounding via GTA1 server.

    No screen preprocessing — the raw screenshot is sent directly to the LLM.
    Positional actions carry a natural-language ``target`` description that the
    GTA1 model resolves to pixel coordinates.
    """

    def __init__(self, gta1_client: GTA1Client) -> None:
        self._client = gta1_client
        self._last_grounding_events: List[Dict[str, Any]] = []

    @property
    def last_grounding_events(self) -> List[Dict[str, Any]]:
        return self._last_grounding_events

    @property
    def name(self) -> str:
        return "gta1"

    @property
    def element_reference_hint(self) -> str:
        return (
            "Describe the UI element you want to interact with in plain English "
            "using the `target` parameter "
            "(e.g. 'the blue Submit button near the bottom of the form')."
        )

    def preprocess(
        self,
        raw_b64: str,
        screen_width: int,
        screen_height: int,
        resized_width: int,
        resized_height: int,
    ) -> ScreenData:
        # Pass-through: GTA1 works on the raw screenshot.
        return ScreenData(
            raw_image_b64=raw_b64,
            display_image_b64=raw_b64,
            elements=[],
            screen_width=screen_width,
            screen_height=screen_height,
            resized_width=resized_width,
            resized_height=resized_height,
        )

    def get_tools(self) -> List[dict]:
        return GTA1_COMPUTER_TOOLS

    def _resolve_positional(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        screen_data: ScreenData,
    ) -> List[int]:
        target = arguments.get("target")
        if not target:
            raise ValueError(f"'{tool_name}' requires target")

        coord = self._resolve_coordinate(screen_data.raw_image_b64, target)
        if coord is None:
            self._last_grounding_events = [{
                "instruction": target,
                "coordinate": None,
                "annotated_image_b64": "",
                "success": False,
            }]
            raise ValueError(f"GTA1 failed to resolve target: {target!r}")

        rx, ry = coord
        annotated = _draw_crosshair(screen_data.raw_image_b64, rx, ry)
        self._last_grounding_events = [{
            "instruction": target,
            "coordinate": [rx, ry],
            "annotated_image_b64": annotated,
            "success": True,
        }]

        rw = screen_data.resized_width or screen_data.screen_width
        rh = screen_data.resized_height or screen_data.screen_height
        sx = round(rx * screen_data.screen_width / rw) if rw else rx
        sy = round(ry * screen_data.screen_height / rh) if rh else ry
        return [sx, sy]

    def _resolve_coordinate(
        self, image_b64: str, instruction: str
    ) -> Optional[List[int]]:
        if not image_b64 or not instruction:
            return None
        try:
            result = self._client.ground(image_b64, instruction)
            return [result["x"], result["y"]]
        except Exception as exc:
            logger.warning("GTA1 grounding failed for %r: %s", instruction, exc)
            return None


__all__ = [
    "ScreenData",
    "GroundingStrategy",
    "OmniParserGrounding",
    "GTA1Grounding",
]
