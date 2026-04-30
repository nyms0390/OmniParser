"""
PaddleOCR GPU API client for remote text recognition.
"""

from typing import Any, Dict, List
import logging

import requests

from .base import BaseServiceClient

logger = logging.getLogger(__name__)


class PaddleOCRClient(BaseServiceClient):
    """GPU PaddleOCR 3.x API client.

    Communicates with a remote PaddleOCR 3.x server that provides GPU-accelerated
    text recognition. Exposes two endpoints on the same base URL:
    - PP-OCRv5 (recognize): flat text + coordinates, used by OmniParser OCR.
    - PaddleOCR-VL (recognize_vl): structured document parsing, used by table scan.
    """

    V5_ENDPOINT = "infer/v5/raw"
    VL_ENDPOINT = "infer/vl/html"

    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 60):
        """Initialize PaddleOCR API client.

        Args:
            base_url: Base URL of PaddleOCR API server (default: localhost:8001)
            timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(base_url, timeout)
        logger.info(f"Initialized PaddleOCR GPU API client at {base_url}")

    @property
    def probe_endpoint(self) -> str:
        """Endpoint for health check."""
        return "probe"

    def _post_image(self, endpoint: str, image: bytes) -> requests.Response:
        """POST a PNG image to the given endpoint and return the raw response."""
        files = {'file': ('image.png', image, 'image/png')}
        return self._make_request("POST", endpoint, json_data=None, files=files)

    def recognize(self, image: bytes) -> List[Dict[str, Any]]:
        """Recognize text in image via PP-OCRv5 endpoint.

        Returns one entry per page (a single image yields a one-element list,
        a multi-page PDF yields one entry per page). Each entry is the
        ``OCRResult.json`` dict produced by ``PaddleOCR.predict()``: the
        recognized text, polygons, and scores live under the ``res`` key
        (``rec_texts``, ``rec_polys``, ``rec_scores``, ...).

        Args:
            image: Image as PNG bytes

        Returns:
            ``[{"res": {"rec_texts": [...], "rec_polys": [...], ...}, ...}, ...]``

        Raises:
            Exception: If API call fails
        """
        return self._post_image(self.V5_ENDPOINT, image).json()

    def recognize_vl(self, image: bytes) -> str:
        """Parse document via PaddleOCR-VL and return server-rendered HTML.

        Backed by ``PaddleOCRVL.save_to_html()`` on the server. Used by the
        table-scan path: callers feed the HTML into the same extraction
        pipeline as the browser-DevTools path.

        Args:
            image: Image as PNG bytes

        Returns:
            The full HTML document as a string.

        Raises:
            Exception: If API call fails
        """
        return self._post_image(self.VL_ENDPOINT, image).text
