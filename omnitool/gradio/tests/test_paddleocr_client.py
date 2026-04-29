"""
Tests for PaddleOCRClient — verifies endpoint routing and payload shape
for the PP-OCRv5 and PaddleOCR-VL paths.

The HTTP layer (BaseServiceClient._make_request) is patched on the instance.
"""

from unittest.mock import patch

import pytest

from omnitool.gradio.clients.external.paddleocr import PaddleOCRClient


@pytest.fixture
def client():
    return PaddleOCRClient(base_url="http://test:8001", timeout=5)


class TestEndpointConstants:
    def test_v5_and_vl_endpoints_differ(self):
        assert PaddleOCRClient.V5_ENDPOINT == "infer/v5/raw"
        assert PaddleOCRClient.VL_ENDPOINT == "infer/vl/html"
        assert PaddleOCRClient.V5_ENDPOINT != PaddleOCRClient.VL_ENDPOINT

    def test_probe_endpoint_is_health(self, client):
        assert client.probe_endpoint == "health"


class TestRecognizeV5:
    def test_posts_to_v5_endpoint_with_multipart_image(self, client):
        expected = {"coordinates": [[0, 0], [10, 10]], "text": ["hi"], "confidence": [0.9]}
        with patch.object(client, "_make_request", return_value=expected) as mock_req:
            result = client.recognize(b"\x89PNG\r\n\x1a\nfake")

        assert result == expected
        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        assert args == ("POST", "infer/v5/raw")
        assert kwargs["json_data"] is None
        assert kwargs["files"] == {"file": ("image.png", b"\x89PNG\r\n\x1a\nfake", "image/png")}

    def test_propagates_request_failure(self, client):
        with patch.object(client, "_make_request", side_effect=Exception("boom")):
            with pytest.raises(Exception, match="boom"):
                client.recognize(b"data")


class TestRecognizeVL:
    def test_posts_to_vl_endpoint_with_multipart_image(self, client):
        expected = {"html": "<html><body><table><tr><td>a</td></tr></table></body></html>"}
        with patch.object(client, "_make_request", return_value=expected) as mock_req:
            result = client.recognize_vl(b"\x89PNG\r\n\x1a\nfake")

        assert result == expected
        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        assert args == ("POST", "infer/vl/html")
        assert kwargs["json_data"] is None
        assert kwargs["files"] == {"file": ("image.png", b"\x89PNG\r\n\x1a\nfake", "image/png")}

    def test_propagates_request_failure(self, client):
        with patch.object(client, "_make_request", side_effect=Exception("boom")):
            with pytest.raises(Exception, match="boom"):
                client.recognize_vl(b"data")
