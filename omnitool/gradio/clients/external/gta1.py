"""
GTA1 grounding model service client.

Sends a raw screenshot + natural-language instruction to the GTA1 server
and returns the resolved pixel coordinate.
"""

import logging
import os
from typing import Any, Dict

from .base import BaseServiceClient

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:8002"


class GTA1Client(BaseServiceClient):
    """HTTP client for the GTA1 GUI grounding server.

    The server accepts a base64-encoded screenshot and a natural-language
    instruction describing a UI element, and returns the ``(x, y)`` pixel
    coordinate of that element in the *original* image space.

    Example::

        client = GTA1Client()
        result = client.ground(image_b64, "click the Firefox icon in the taskbar")
        x, y = result["x"], result["y"]
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 30,
    ):
        """Initialize the GTA1 client.

        Args:
            base_url: URL of the GTA1 server.  Falls back to the ``GTA1_URL``
                environment variable, then ``http://localhost:8002``.
            timeout: Request timeout in seconds.
        """
        resolved = base_url or os.environ.get("GTA1_URL", _DEFAULT_URL)
        super().__init__(resolved, timeout)
        logger.info("Initialized GTA1Client at %s", resolved)

    @property
    def probe_endpoint(self) -> str:
        return "probe"

    def ground(self, image_base64: str, instruction: str) -> Dict[str, Any]:
        """Resolve a natural-language instruction to pixel coordinates.

        Args:
            image_base64: Base64-encoded PNG/JPEG screenshot.
            instruction: Natural-language description of the target element,
                e.g. ``"double-click the Firefox browser icon in the taskbar"``.

        Returns:
            Dict with keys:
                - ``x`` (int): Pixel x-coordinate in the original image.
                - ``y`` (int): Pixel y-coordinate in the original image.
                - ``raw_output`` (str): Raw model output string.

        Raises:
            Exception: If the HTTP request fails or the server returns an error.
        """
        result = self._make_request(
            "POST",
            "ground",
            json_data={
                "image_base64": image_base64,
                "instruction": instruction,
            },
        )
        return {
            "x": int(result["x"]),
            "y": int(result["y"]),
            "raw_output": result.get("raw_output", ""),
        }
