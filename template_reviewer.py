"""
YAML Template Reviewer - LLM-powered template validation and correction.

Uploads a YAML task template, sends it alongside TEMPLATE_GUIDE.md to an
Azure OpenAI model, and displays the corrected template with highlighted diffs.

Usage:
    python template_reviewer.py
    python template_reviewer.py --azure-endpoint https://xxx.openai.azure.com/ --model gpt-4.1
"""

import argparse
import difflib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import yaml

from omnitool.gradio.clients.llm import get_llm_client
from omnitool.gradio.config import setup_logging


logger = logging.getLogger("template_reviewer")

_previous_download_path: Optional[str] = None

TEMPLATE_GUIDE_PATH = Path(__file__).parent / "TEMPLATE_GUIDE.md"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_PORT = 7862

# Structured output schema — the model must return exactly this JSON shape.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "template_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "corrected_yaml": {
                    "type": "string",
                    "description": "The full corrected YAML template, with no surrounding backticks or fences.",
                },
                "changes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of corrections made. Empty if no changes were needed.",
                },
            },
            "required": ["corrected_yaml", "changes"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """\
You are a YAML task template validator and corrector. You will be given a template guide \
followed by a YAML template to review. Correct the template so it fully complies with the guide.

Rules:
- Preserve the user's intent and content. Only change what violates the guide.
- If the YAML is syntactically malformed, fix it while preserving as much original content as possible.
- Do not add, remove, or alter steps unless they violate a guide rule.
- The corrected_yaml field must contain valid YAML that passes yaml.safe_load() without error.
- The changes field must list each correction as a separate string. If no changes were needed, return an empty list.
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YAML Template Reviewer")
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        help="Azure OpenAI endpoint URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("AZURE_OPENAI_MODEL", DEFAULT_MODEL),
        help=f"Azure model deployment name (default: {DEFAULT_MODEL})",
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
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_template_guide() -> str:
    if not TEMPLATE_GUIDE_PATH.exists():
        raise FileNotFoundError(f"TEMPLATE_GUIDE.md not found at {TEMPLATE_GUIDE_PATH}")
    return TEMPLATE_GUIDE_PATH.read_text(encoding="utf-8")


def on_file_upload(file) -> str:
    """Read uploaded YAML file and return its text content."""
    if file is None:
        return ""
    try:
        path = Path(file.name) if hasattr(file, "name") else Path(file)
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"# Error reading file: {e}"


def build_user_message(guide_text: str, yaml_text: str) -> str:
    return (
        "## Template Guide\n\n"
        f"{guide_text}\n\n"
        "---\n\n"
        "## Template to Review\n\n"
        f"{yaml_text}"
    )


def build_html_diff(original: str, corrected: str) -> str:
    """Return an HTML diff table comparing original and corrected YAML."""
    original_lines = original.splitlines()
    corrected_lines = corrected.splitlines()
    differ = difflib.HtmlDiff(wrapcolumn=100)
    table = differ.make_table(
        original_lines,
        corrected_lines,
        fromdesc="Original",
        todesc="Corrected",
        context=True,
        numlines=3,
    )
    return (
        "<style>"
        ".diff { font-family: monospace; font-size: 13px; border-collapse: collapse; width: 100%; }"
        ".diff td { padding: 2px 8px; vertical-align: top; white-space: pre-wrap; word-break: break-all; }"
        ".diff th { padding: 4px 8px; background: #f0f0f0; text-align: left; }"
        ".diff_add { background-color: #d4edda; }"
        ".diff_chg { background-color: #fff3cd; }"
        ".diff_sub { background-color: #f8d7da; }"
        ".diff_next { background-color: #f0f0f0; color: #888; }"
        "</style>"
        + table
    )


def review_template(
    original_text: str,
    azure_endpoint: str,
    model_name: str,
    guide_text: str,
) -> Tuple[str, str, str, str, Optional[str]]:
    """Call the LLM and return (corrected_yaml, changes, diff_html, status, download_path)."""
    _cleanup_previous_download()

    if not original_text or not original_text.strip():
        return "", "", "", "Please upload a YAML file first.", None

    if not azure_endpoint or not azure_endpoint.strip():
        return "", "", "", "Azure endpoint is required. Set it in the configuration panel or via --azure-endpoint.", None

    # Warn but proceed if original YAML is malformed
    status_warnings = []
    try:
        yaml.safe_load(original_text)
    except yaml.YAMLError as e:
        status_warnings.append(
            f"Warning: original YAML has syntax errors ({e}). Sending to LLM for correction."
        )

    logger.info(f"Calling Azure OpenAI model '{model_name}' at {azure_endpoint}...")
    try:
        client = get_llm_client("azure", model_name, azure_endpoint=azure_endpoint.strip())
        user_message = build_user_message(guide_text, original_text)
        response_text, metadata = client.generate(
            [{"role": "user", "content": user_message}],
            system_prompt=SYSTEM_PROMPT,
            response_format=RESPONSE_FORMAT,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "", "", "", f"LLM call failed: {e}", None

    logger.info(f"LLM call complete. Tokens: {metadata.get('tokens', '?')}")

    # Parse structured JSON response
    try:
        data = json.loads(response_text)
        corrected_yaml = data["corrected_yaml"]
        changes_list = data["changes"]
    except (json.JSONDecodeError, KeyError) as e:
        return response_text, "", "", f"Failed to parse structured response: {e}", None

    changes_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(changes_list)) if changes_list else "No changes required."

    # Validate corrected YAML
    diff_html = ""
    try:
        yaml.safe_load(corrected_yaml)
    except yaml.YAMLError as e:
        status_warnings.append(f"Warning: corrected YAML has syntax errors ({e}). Diff skipped.")
    else:
        if corrected_yaml.strip() == original_text.strip():
            diff_html = "<p style='color: #555; font-family: monospace;'>No changes were necessary — template already complies with the guide.</p>"
        else:
            diff_html = build_html_diff(original_text, corrected_yaml)

    # Write corrected YAML to a temp file for download
    download_path = _write_download_file(corrected_yaml) if corrected_yaml.strip() else None

    status = "\n".join(status_warnings) if status_warnings else "Review complete."
    download_update = gr.update(value=download_path, visible=download_path is not None)
    return corrected_yaml, changes_text, diff_html, status, download_update


def _cleanup_previous_download() -> None:
    global _previous_download_path
    if _previous_download_path:
        try:
            os.unlink(_previous_download_path)
        except OSError:
            pass
        _previous_download_path = None


def _write_download_file(text: str) -> str:
    """Write text to a new temp file and return its path."""
    global _previous_download_path
    tmp = tempfile.NamedTemporaryFile(
        mode="w", prefix="corrected_template_", suffix=".yaml",
        delete=False, encoding="utf-8",
    )
    with tmp:
        tmp.write(text)
    _previous_download_path = tmp.name
    return tmp.name


def update_download_file(text: str):
    """Rewrite the temp download file when the user finishes editing the corrected YAML."""
    _cleanup_previous_download()
    if not text or not text.strip():
        return gr.update(value=None, visible=False)
    path = _write_download_file(text)
    return gr.update(value=path, visible=True)


def build_ui(default_endpoint: str, default_model: str, guide_text: str) -> gr.Blocks:
    with gr.Blocks(title="YAML Template Reviewer") as demo:
        gr.Markdown(
            "# YAML Template Reviewer\n"
            "Upload a task template YAML. The LLM will correct it against the guide and highlight changes."
        )

        with gr.Accordion("Azure Configuration", open=not default_endpoint):
            azure_endpoint_input = gr.Textbox(
                label="Azure OpenAI Endpoint",
                value=default_endpoint,
                placeholder="https://your-resource.openai.azure.com/",
            )
            model_name_input = gr.Textbox(
                label="Model Deployment Name",
                value=default_model,
                placeholder=DEFAULT_MODEL,
            )

        with gr.Row():
            yaml_upload = gr.File(
                label="Upload YAML Template",
                file_count="single",
                file_types=[".yaml", ".yml"],
                scale=3,
            )
            review_btn = gr.Button("Review Template", variant="primary", scale=1)

        original_yaml = gr.Textbox(
            label="Original YAML",
            lines=20,
            interactive=False,
            show_copy_button=True,
        )

        with gr.Row():
            corrected_yaml = gr.Textbox(
                label="Corrected YAML (editable)",
                lines=20,
                interactive=True,
                show_copy_button=True,
            )
            changes_text = gr.Textbox(
                label="Changes Made",
                lines=20,
                interactive=False,
            )

        re_review_btn = gr.Button("Re-review Edited YAML", variant="secondary")

        diff_html = gr.HTML(label="Diff View")

        download_file = gr.File(
            label="Download Corrected YAML",
            interactive=False,
            visible=False,
        )

        status_box = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
        )

        yaml_upload.change(
            fn=on_file_upload,
            inputs=[yaml_upload],
            outputs=[original_yaml],
        )

        review_btn.click(
            fn=lambda orig, endpoint, model: review_template(orig, endpoint, model, guide_text),
            inputs=[original_yaml, azure_endpoint_input, model_name_input],
            outputs=[corrected_yaml, changes_text, diff_html, status_box, download_file],
        )

        corrected_yaml.blur(
            fn=update_download_file,
            inputs=[corrected_yaml],
            outputs=[download_file],
        )

        def _re_review(edited, endpoint, model):
            if not edited or not edited.strip():
                return "", "", "", "No corrected YAML to re-review. Run an initial review first.", gr.update(value=None, visible=False)
            return review_template(edited, endpoint, model, guide_text)

        re_review_btn.click(
            fn=_re_review,
            inputs=[corrected_yaml, azure_endpoint_input, model_name_input],
            outputs=[corrected_yaml, changes_text, diff_html, status_box, download_file],
        )

    return demo


def main() -> None:
    args = parse_arguments()
    setup_logging("template_reviewer", level=args.log_level, log_file="template_reviewer.log")

    guide_text = load_template_guide()
    logger.info(f"Loaded TEMPLATE_GUIDE.md ({len(guide_text)} chars)")

    demo = build_ui(args.azure_endpoint, args.model, guide_text)

    logger.info(f"Launching at {args.server_name}:{args.server_port}")
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
