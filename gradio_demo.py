"""
OmniParser Gradio Demo - HTTP Client to OmniParser Server.

This demo provides a web interface for testing the OmniParser server API.
Requires OmniParser server running (default: http://127.0.0.1:8000).

Usage:
    python gradio_demo.py --omniparser-url http://127.0.0.1:8000
"""

import argparse
import base64
import io
import logging
from typing import Any, Dict, Tuple

import gradio as gr
import requests
from PIL import Image

from omnitool.gradio.clients import OmniParserClient
from omnitool.gradio.config import setup_logging


# Will be initialized in main()
logger = None

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OmniParser Gradio Demo - HTTP Client"
    )
    parser.add_argument(
        "--omniparser-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="URL of OmniParser server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share Gradio interface via public link",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server name for Gradio interface",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7861,
        help="Server port for Gradio interface",
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
        help="Optional log file path (default: gradio_demo.log)",
    )
    return parser.parse_args()


def create_process_function(
    omniparser_client: OmniParserClient,
) -> callable:
    """Create the process function with OmniParser client."""
    
    async def process(
        image_input: Image.Image,
        box_threshold: float,
        iou_threshold: float,
        use_paddleocr: bool,
        imgsz: int,
    ) -> Tuple[Image.Image, str]:
        """
        Process image via OmniParser API.
        
        Args:
            image_input: PIL Image to process
            box_threshold: Confidence threshold for box detection
            iou_threshold: IOU threshold for filtering overlapping boxes
            use_paddleocr: Whether to use PaddleOCR (UI control only, server-side parameter)
            imgsz: Image size for detection model (UI control only, server-side parameter)
            
        Returns:
            Tuple of (labeled_image, formatted_content)
        """
        try:
            logger.info("Processing image via OmniParser API...")
            
            # Convert PIL image to base64
            img_bytes = io.BytesIO()
            image_input.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            base64_image = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
            
            # Call API
            logger.info(f"Calling API: {omniparser_client.base_url}/parse/")
            response = omniparser_client.parse_screenshot(base64_image)
            
            # Extract and decode response
            labeled_img_b64 = response["labeled_screenshot_base64"]
            labeled_img = Image.open(
                io.BytesIO(base64.b64decode(labeled_img_b64))
            )
            
            # Format parsed content
            parsed_list: list = response.get("parsed_content_list", [])
            latency: float = response.get("latency", 0.0)
            
            formatted_lines = []
            for i, item in enumerate(parsed_list):
                item_type = item.get("type", "unknown").capitalize()
                content = item.get("content", "N/A")
                formatted_lines.append(f"{item_type} {i}: {content}")
            
            formatted_content = "\n".join(formatted_lines)
            logger.info(f"Parsing completed in {latency:.2f}s. Found {len(parsed_list)} elements.")
            
            return labeled_img, formatted_content
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            error_msg = f"Error: {str(e)}\n\nEnsure OmniParser server is running at {omniparser_client.base_url}"
            return Image.new("RGB", (100, 100)), error_msg
    
    return process


def check_server_health(base_url: str) -> bool:
    """
    Check if OmniParser server is running and healthy.
    
    Args:
        base_url: Base URL of OmniParser server
        
    Returns:
        True if server is healthy, False otherwise
    """
    try:
        health_url = f"{base_url.rstrip('/')}/health/"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                logger.info(f"✅ Server health check passed: {health_data}")
                return True
            else:
                logger.warning(f"⚠️ Server returned non-healthy status: {health_data}")
                return False
        else:
            logger.warning(f"⚠️ Server health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Cannot connect to OmniParser server at {base_url}")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Make sure server is running: python -m omniparserserver")
        return False
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ Server health check timed out at {base_url}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during health check: {str(e)}")
        return False


def main() -> None:
    """Main function to run Gradio demo."""
    global logger
    
    args = parse_arguments()
    
    # Setup logging (FIRST before any other code)
    log_file = args.log_file or "gradio_demo.log"
    logger = setup_logging("gradio_demo", level=args.log_level, log_file=log_file)
    
    # Initialize OmniParser client
    logger.info(f"Connecting to OmniParser server at {args.omniparser_url}...")
    omniparser_client = OmniParserClient(args.omniparser_url)
    
    # Test server health
    server_healthy = check_server_health(args.omniparser_url)
    
    if server_healthy:
        logger.info("OmniParser server is ready for use")
    else:
        logger.warning("OmniParser server health check failed. Server may not be running.")
        logger.warning("The app will attempt to use the server anyway, but requests may fail.")
    
    # Create process function
    process_fn = create_process_function(omniparser_client)
    
    # Build Gradio UI
    with gr.Blocks(title="OmniParser Demo") as demo:
        gr.Markdown(MARKDOWN)
        with gr.Row():
            with gr.Column():
                image_input_component = gr.Image(
                    type="pil", label="Upload image"
                )
                # set the threshold for removing the bounding boxes with low confidence, default is 0.05
                box_threshold_component = gr.Slider(
                    label="Box Threshold",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=0.05,
                )
                # set the threshold for removing the bounding boxes with large overlap, default is 0.1
                iou_threshold_component = gr.Slider(
                    label="IOU Threshold",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=0.1,
                )
                use_paddleocr_component = gr.Checkbox(
                    label="Use PaddleOCR", value=True
                )
                imgsz_component = gr.Slider(
                    label="Icon Detect Image Size",
                    minimum=640,
                    maximum=1920,
                    step=32,
                    value=640,
                )
                submit_button_component = gr.Button(
                    value="Submit", variant="primary"
                )
            with gr.Column():
                image_output_component = gr.Image(
                    type="pil", label="Image Output"
                )
                text_output_component = gr.Textbox(
                    label="Parsed screen elements",
                    placeholder="Text Output",
                )

        submit_button_component.click(
            fn=process_fn,
            inputs=[
                image_input_component,
                box_threshold_component,
                iou_threshold_component,
                use_paddleocr_component,
                imgsz_component,
            ],
            outputs=[image_output_component, text_output_component],
        )

    # Launch demo
    logger.info(
        f"Launching Gradio interface at {args.server_name}:{args.server_port}"
    )
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
