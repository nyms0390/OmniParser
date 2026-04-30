"""FastAPI inference server for PaddleOCR (PP-OCRv5 + PaddleOCR-VL).

Designed to run in its own container, fronting two PaddleOCR pipelines:
- PP-OCRv5      → text detection + recognition (flat output)
- PaddleOCR-VL  → document parsing (structured JSON + rendered HTML)

Endpoints (consumed by ``omnitool.gradio.clients.external.paddleocr.PaddleOCRClient``):
    GET  /probe         — liveness probe
    POST /infer/v5/raw  — multipart 'file' (image or PDF) → structured JSON pages
    POST /infer/vl/raw  — multipart 'file' (image or PDF) → structured JSON pages
    POST /infer/vl/html — multipart 'file' (image or PDF) → server-rendered HTML

Supported input formats: .png .jpg .jpeg .bmp .tiff .tif .webp .pdf

Environment overrides:
    PADDLE_DEVICE            (default 'gpu:0')
    PADDLE_LANG              (default 'en')
    VL_REC_SERVER_URL        (default 'http://vllm:8000/v1')
    VL_REC_API_MODEL_NAME    (default 'PaddleOCR-VL-1.5-0.9B')
    LAYOUT_MODEL_DIR         (default '/root/.cache/huggingface/PaddlePaddle/PP-DocLayoutV2')
Launch:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("paddleocr_app")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".pdf"}


# ---------------------------------------------------------------------------
# Engine builders
# ---------------------------------------------------------------------------

_v5_engine: Any = None
_vl_engine: Any = None


def _build_v5_engine(device: str, lang: str):
    from paddleocr import PaddleOCR  # type: ignore[import-untyped]

    return PaddleOCR(
        device=device,
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _build_vl_engine(device: str):
    from paddleocr import PaddleOCRVL  # type: ignore[import-untyped]

    return PaddleOCRVL(
        device=device,
        vl_rec_backend="vllm-server",
        vl_rec_server_url=os.environ.get("VL_REC_SERVER_URL", "http://vllm:8000/v1"),
        vl_rec_api_model_name=os.environ.get("VL_REC_API_MODEL_NAME", "PaddleOCR-VL-1.5-0.9B"),
        layout_detection_model_name="PP-DocLayoutV2",
        layout_detection_model_dir=os.environ.get("LAYOUT_MODEL_DIR", "/root/.cache/huggingface/PaddlePaddle/PP-DocLayoutV2"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _v5_engine, _vl_engine
    device = os.environ.get("PADDLE_DEVICE", "gpu:0")
    lang = os.environ.get("PADDLE_LANG", "ch")

    logger.info("Building PP-OCRv5 engine (device=%s, lang=%s)", device, lang)
    _v5_engine = _build_v5_engine(device, lang)

    logger.info("Building PaddleOCR-VL engine (device=%s)", device)
    _vl_engine = _build_vl_engine(device)

    logger.info("Engines ready.")
    yield
    _v5_engine = None
    _vl_engine = None


app = FastAPI(title="PaddleOCR Inference Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(file: UploadFile, content: bytes, tmp: str) -> Path:
    """Write upload bytes to a temp file, preserving the original extension."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    path = Path(tmp) / f"input{suffix}"
    path.write_bytes(content)
    return path


def _jsonable(obj: Any) -> Any:
    """Recursively coerce PaddleOCR result trees into JSON-safe primitives.

    PaddleOCR's ``result.json`` mostly returns plain dicts, but numpy arrays,
    numpy scalars, and ``Path`` instances occasionally leak through and would
    crash the default JSON encoder.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bytes, Path)):
        return str(obj)
    return obj


def _md_to_html(md: str) -> str:
    """Minimal Markdown-to-HTML converter (no external dependencies).

    Handles the subset produced by PaddleOCR-VL: ATX headings, fenced code
    blocks, GFM tables, horizontal rules, images, bold, italic, inline code,
    paragraphs.  Mirrors the implementation in paddleocr_infer.py.
    """
    lines = md.splitlines()
    html_parts: list[str] = []
    i = 0

    _BLOCK_HTML_RE = re.compile(
        r'^\s*</?(?:div|figure|section|article|blockquote|ul|ol|li|dl|dt|dd'
        r'|header|footer|nav|aside|main|details|summary)[^>]*>',
        re.IGNORECASE,
    )

    def _inline(text: str) -> str:
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img alt="\1" src="\2">', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        return text

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            lang = line.strip()[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            lang_attr = f' class="language-{lang}"' if lang else ""
            escaped = "\n".join(code_lines).replace("&", "&amp;").replace("<", "&lt;")
            html_parts.append(f"<pre><code{lang_attr}>{escaped}</code></pre>")
            i += 1
            continue

        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            html_parts.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if re.match(r'^(\*{3,}|-{3,}|_{3,})\s*$', line):
            html_parts.append("<hr>")
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r'^\|?\s*[-:]+', lines[i + 1]):
            def _parse_row(row: str) -> list[str]:
                return [c.strip() for c in row.strip().strip("|").split("|")]
            headers = _parse_row(line)
            i += 2
            header_html = "".join(f"<th>{_inline(h)}</th>" for h in headers)
            rows_html: list[str] = []
            while i < len(lines) and "|" in lines[i]:
                row_html = "".join(f"<td>{_inline(c)}</td>" for c in _parse_row(lines[i]))
                rows_html.append(f"<tr>{row_html}</tr>")
                i += 1
            html_parts.append(
                f"<table>\n<thead><tr>{header_html}</tr></thead>\n"
                f"<tbody>\n{''.join(rows_html)}\n</tbody>\n</table>"
            )
            continue

        if line.strip() == "":
            html_parts.append("")
            i += 1
            continue

        if _BLOCK_HTML_RE.match(line):
            html_parts.append(line)
            i += 1
            continue

        para_lines: list[str] = []
        while i < len(lines):
            l = lines[i]
            if (
                l.strip() == ""
                or re.match(r'^#{1,6}\s', l)
                or l.strip().startswith("```")
                or re.match(r'^(\*{3,}|-{3,}|_{3,})\s*$', l)
                or _BLOCK_HTML_RE.match(l)
                or ("|" in l and i + 1 < len(lines) and re.match(r'^\|?\s*[-:]+', lines[i + 1]))
            ):
                break
            para_lines.append(_inline(l))
            i += 1
        if para_lines:
            html_parts.append(f"<p>{'<br>'.join(para_lines)}</p>")

    body = "\n".join(html_parts)
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        "<style>\n"
        "  body { font-family: sans-serif; max-width: 960px; margin: 2em auto; }\n"
        "  table { border-collapse: collapse; width: 100%; }\n"
        "  th, td { border: 1px solid #ccc; padding: 6px 10px; }\n"
        "  th { background: #f0f0f0; }\n"
        "  pre { background: #f8f8f8; padding: 1em; overflow-x: auto; }\n"
        "  img { max-width: 100%; }\n"
        "</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/probe")
async def probe():
    return {
        "status": "ok",
        "v5_loaded": _v5_engine is not None,
        "vl_loaded": _vl_engine is not None,
    }


@app.post("/infer/v5/raw")
async def infer_v5_raw(file: UploadFile = File(...)):
    if _v5_engine is None:
        raise HTTPException(status_code=503, detail="PP-OCRv5 engine not loaded")

    content = await file.read()
    with tempfile.TemporaryDirectory() as tmp:
        input_path = _save_upload(file, content, tmp)
        results = [r.json for r in _v5_engine.predict(input=str(input_path))]

    return JSONResponse(content=_jsonable(results))


@app.post("/infer/vl/raw")
async def infer_vl_raw(file: UploadFile = File(...)):
    if _vl_engine is None:
        raise HTTPException(status_code=503, detail="PaddleOCR-VL engine not loaded")

    content = await file.read()
    with tempfile.TemporaryDirectory() as tmp:
        input_path = _save_upload(file, content, tmp)
        results = [r.json for r in _vl_engine.predict(input=str(input_path))]

    return JSONResponse(content=_jsonable(results))


@app.post("/infer/vl/html")
async def infer_vl_html(file: UploadFile = File(...)):
    if _vl_engine is None:
        raise HTTPException(status_code=503, detail="PaddleOCR-VL engine not loaded")

    content = await file.read()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = _save_upload(file, content, tmp)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        for res in _vl_engine.predict(input=str(input_path)):
            if res.save_to_html(save_path=str(out_dir)) is None:
                # save_to_html() returns None for VL result types that don't
                # support direct HTML export; fall back via markdown.
                res.save_to_markdown(save_path=str(out_dir))
                for md_path in out_dir.glob("*.md"):
                    md_path.with_suffix(".html").write_text(
                        _md_to_html(md_path.read_text(encoding="utf-8")),
                        encoding="utf-8",
                    )

        html_files = sorted(out_dir.glob("*.html"))
        if not html_files:
            raise HTTPException(status_code=500, detail="VL engine produced no HTML output")
        html = "\n".join(f.read_text(encoding="utf-8") for f in html_files)

    return HTMLResponse(html)

