"""
OmniParser Server - FastAPI endpoint for image parsing.

Usage:
    python -m omniparserserver \
        --som_model_path ../../weights/icon_detect/model.pt \
        --caption_model_name florence2 \
        --caption_model_path ../../weights/icon_caption_florence \
        --device cuda \
        --BOX_TRESHOLD 0.05

Environment:
    Package must be installed: pip install -e /path/to/OmniParser
"""

import argparse
import logging
import time
from typing import Any, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from omnitool.gradio.config import setup_logging

# Import from util module (non-refactored legacy parser logic)
import sys
from pathlib import Path

# Add util module to path for legacy parser
util_dir = Path(__file__).parent.parent.parent / "util"
sys.path.insert(0, str(util_dir))

from omniparser import Omniparser

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for OmniParser server."""
    parser = argparse.ArgumentParser(description="OmniParser API Server")
    parser.add_argument(
        "--som_model_path",
        type=str,
        default="../../weights/icon_detect/model.pt",
        help="Path to the SOM (Set-of-Marks) model",
    )
    parser.add_argument(
        "--caption_model_name",
        type=str,
        default="florence2",
        help="Name of the caption model",
    )
    parser.add_argument(
        "--caption_model_path",
        type=str,
        default="../../weights/icon_caption_florence",
        help="Path to the caption model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (cpu or cuda)",
    )
    parser.add_argument(
        "--BOX_TRESHOLD",
        type=float,
        default=0.05,
        help="Threshold for box detection",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path (default: omniparserserver.log)",
    )
    return parser.parse_args()


# Initialize parser and configuration
args = parse_arguments()
config = vars(args)

# Extract logging args and setup logging
log_level = args.log_level
log_file = args.log_file or "omniparserserver.log"
logger = setup_logging("omniparserserver", level=log_level, log_file=log_file)

# Remove log_level and log_file from config (not needed for Omniparser)
config.pop("log_level", None)
config.pop("log_file", None)

# Initialize FastAPI app
app = FastAPI(
    title="OmniParser API",
    description="API for parsing UI screenshots and extracting semantic information",
    version="1.0.0",
)

# Initialize OmniParser
omniparser = Omniparser(config)


# ============================================================================
# Request/Response Models
# ============================================================================


class ParseRequest(BaseModel):
    """Request model for parsing an image."""

    base64_image: str


class ParseResponse(BaseModel):
    """Response model for parsed screenshot."""

    labeled_screenshot_base64: str
    """Base64 encoded screenshot with UI element labels and bounding boxes."""
    
    parsed_content_list: List[dict]
    """List of detected UI elements with metadata:
    Each element is a dict with keys:
    - type: 'text' or 'icon'
    - bbox: [x1, y1, x2, y2] bounding box coordinates
    - interactivity: bool, whether element is interactive
    - content: str (text content) or None (for icons)
    - source: str, source of content identification
    """
    
    latency: float
    """Processing time in seconds."""


# ============================================================================
# API Endpoints
# ============================================================================


@app.post("/parse/", response_model=ParseResponse)
async def parse(parse_request: ParseRequest) -> ParseResponse:
    """
    Parse an image and extract semantic information.

    Args:
        parse_request: Request containing base64-encoded image

    Returns:
        ParseResponse with labeled image and parsed content
    """
    logger.info("Starting image parsing...")
    start = time.time()

    try:
        dino_labled_img, parsed_content_list = omniparser.parse(
            parse_request.base64_image
        )
        latency = time.time() - start
        logger.info(f"Parsing completed in {latency:.2f}s")

        return ParseResponse(
            labeled_screenshot_base64=dino_labled_img,
            parsed_content_list=parsed_content_list,
            latency=latency,
        )
    except Exception as e:
        logger.error(f"Error parsing image: {e}")
        raise


@app.get("/health/")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "OmniParser API"}


if __name__ == "__main__":
    """Run the OmniParser API server."""
    logger.info(f"Starting OmniParser API server on {args.host}:{args.port}")
    uvicorn.run(
        "omniparserserver:app",
        host=args.host,
        port=args.port,
        reload=True,
        log_level="info",
    )