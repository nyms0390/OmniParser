"""OmniParser Server - FastAPI endpoint for image parsing.

This module provides a FastAPI-based HTTP API for parsing UI screenshots
and extracting semantic information using the OmniParser model.

Usage:
    python -m omnitool.omniparserserver \
        --som_model_path ../../weights/icon_detect/model.pt \\
        --caption_model_name florence2 \\
        --caption_model_path ../../weights/icon_caption_florence \\
        --BOX_TRESHOLD 0.05

Environment:
    Package must be installed: pip install -e /path/to/OmniParser

Example:
    Start the server:
        python -m omnitool.omniparserserver --port 8000

    Make a request:
        curl -X POST http://localhost:8000/parse/ \\
             -H "Content-Type: application/json" \\
             -d '{"base64_image": "..."}'
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Add util module to Python path for imports
_util_dir = Path(__file__).parent.parent.parent / "util"
sys.path.insert(0, str(_util_dir))

from omniparser import Omniparser

from omnitool.gradio.config import create_argument_parser, setup_logging



def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments for OmniParser server.

    Extends the shared argument parser from omnitool.gradio.config with
    server-specific model and network configuration options.

    Returns:
        argparse.Namespace: Parsed command-line arguments with all options.

    Raises:
        SystemExit: If required arguments are missing or invalid.
    """
    parser = create_argument_parser()

    # Server-specific model configuration arguments
    parser.add_argument(
        "--som_model_path",
        type=str,
        default="weights/icon_detect/model.pt",
        help="Path to the SOM (Set-of-Marks) YOLO model weights.",
    )
    parser.add_argument(
        "--caption_model_name",
        type=str,
        default="florence2",
        help="Name of the caption model (e.g., 'florence2' or 'blip2').",
    )
    parser.add_argument(
        "--caption_model_path",
        type=str,
        default="weights/icon_caption_florence",
        help="Path or identifier for caption model weights.",
    )
    parser.add_argument(
        "--BOX_TRESHOLD",
        type=float,
        default=0.05,
        help="Confidence threshold for YOLO object detection [0.0-1.0].",
    )

    # Server network configuration arguments
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host address to bind to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number to listen on.",
    )

    return parser.parse_args()


# ============================================================================
# Initialization
# ============================================================================

# Parse command-line arguments and configuration
_args = parse_arguments()
_config = vars(_args)

# Configure logging from settings
_log_level = _args.log_level
_log_file = _args.log_file or "omniparserserver.log"
logger = setup_logging("omniparserserver", level=_log_level, log_file=_log_file)

# Remove logging-specific config keys (not needed for OmniParser)
_config.pop("log_level", None)
_config.pop("log_file", None)

# Initialize FastAPI application
app = FastAPI(
    title="OmniParser API",
    description=(
        "HTTP API for parsing UI screenshots and extracting semantic information."
    ),
    version="1.0.0",
)

# Initialize OmniParser instance with configuration
omniparser = Omniparser(_config)
logger.info("OmniParser initialized successfully")


# ============================================================================
# Request and Response Models
# ============================================================================


class ParseRequest(BaseModel):
    """Request model for image parsing endpoint.

    Attributes:
        base64_image: Base64-encoded image data (PNG, JPEG, etc.).
    """

    base64_image: str


class ParseResponse(BaseModel):
    """Response model for parsed screenshot with detected UI elements.

    Attributes:
        labeled_screenshot_base64: Base64-encoded image with UI element
            labels and bounding boxes overlaid.
        parsed_content_list: List of detected UI elements with metadata.
            Each element dict contains:
            - type: Element type ('text' or 'icon')
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - interactivity: Whether the element is interactive
            - content: Text content or None for icons
            - source: Source of content ('ocr' or 'caption')
        latency: Processing time in seconds.
    """

    labeled_screenshot_base64: str
    parsed_content_list: List[Dict[str, Any]]
    latency: float


# ============================================================================
# API Endpoints
# ============================================================================


@app.post("/parse/", response_model=ParseResponse)
async def parse(parse_request: ParseRequest) -> ParseResponse:
    """Parse an image and extract semantic information.

    Detects interactive UI elements and generates semantic labels using
    the OmniParser model. Returns annotated image and parsed content list.

    Args:
        parse_request: Request containing base64-encoded image.

    Returns:
        ParseResponse: Annotated image and parsed UI elements.

    Raises:
        HTTPException: If image parsing fails.
    """
    logger.info("Starting image parsing...")
    start = time.time()

    try:
        labeled_img, parsed_content = omniparser.parse(
            parse_request.base64_image
        )
        latency = time.time() - start
        logger.info(f"Image parsing completed in {latency:.2f}s")

        return ParseResponse(
            labeled_screenshot_base64=labeled_img,
            parsed_content_list=parsed_content,
            latency=latency,
        )
    except Exception as e:
        logger.error(f"Error during image parsing: {e}", exc_info=True)
        raise


@app.get("/health/")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Dict with service status and name.
    """
    return {"status": "healthy", "service": "OmniParser API"}


if __name__ == "__main__":
    """Run OmniParser HTTP API server."""
    logger.info(
        f"Starting OmniParser API server on http://{_args.host}:{_args.port}"
    )
    uvicorn.run(
        "omnitool.omniparserserver.omniparserserver:app",
        host=_args.host,
        port=_args.port,
        reload=False,
        log_level=_log_level.lower(),
    )
