"""
PaddleOCR batch inference CLI for PDFs.

Recursively processes all PDF files in an input folder using either
PaddleOCR-VL (document parsing) or PP-OCRv5 (text detection + recognition),
writing results into an output folder that mirrors the input directory tree.

Requires:
    pip install "paddleocr[doc-parser]"   # for PaddleOCR-VL + PP-OCRv5
    pip install paddlepaddle-gpu>=3.2.1   # or paddlepaddle for CPU

Usage:
    python -m omnitool.paddleocr_infer --input ./docs --output ./results --engine vl
    python -m omnitool.paddleocr_infer --input ./docs --engine ppocr --lang en
"""

import argparse
import logging
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("paddleocr_infer")


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------

def _build_vl_engine(device: str, **kwargs):
    """Instantiate PaddleOCR-VL pipeline (lazy import)."""
    from paddleocr import PaddleOCRVL  # type: ignore[import-untyped]

    engine = PaddleOCRVL(
        device=device,
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://vllm:8000/v1",
        vl_rec_api_model_name="PaddleOCR-VL-1.5-0.9B",
        layout_detection_model_name="PP-DocLayoutv2",
        layout_detection_model_dir="./models/pp_doclayoutv2",
    )
    logger.info("Initialized PaddleOCR-VL engine (device=%s)", device)
    return engine


def _build_ppocr_engine(device: str, lang: str = "en", **kwargs):
    """Instantiate PP-OCRv5 pipeline (lazy import)."""
    from paddleocr import PaddleOCR  # type: ignore[import-untyped]

    engine = PaddleOCR(
        device=device,
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    logger.info("Initialized PP-OCRv5 engine (device=%s, lang=%s)", device, lang)
    return engine


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def _run_vl(engine, pdf_path: Path, out_dir: Path, *, merge_tables: bool, relevel_titles: bool):
    """Run PaddleOCR-VL inference on a single PDF and save outputs."""
    results = list(engine.predict(input=str(pdf_path)))

    if merge_tables or relevel_titles:
        results = list(engine.restructure_pages(
            results,
            merge_tables=merge_tables,
            relevel_titles=relevel_titles,
            concatenate_pages=True,
        ))

    for res in results:
        res.save_to_json(save_path=str(out_dir))
        res.save_to_markdown(save_path=str(out_dir))

        # save_to_html() is not supported on PaddleOCRVLPagesResult (multi-page /
        # restructured); it logs a warning and returns None.  Fall back to converting
        # the markdown we already saved.
        html_result = res.save_to_html(save_path=str(out_dir))
        if html_result is None:
            _convert_md_to_html(out_dir)

    return len(results)


def _run_ppocr(engine, pdf_path: Path, out_dir: Path):
    """Run PP-OCRv5 inference on a single PDF and save res_img only."""
    results = list(engine.predict(input=str(pdf_path)))

    for i, res in enumerate(results):
        res.save_to_json(save_path=str(out_dir))
        _save_ppocr_res_img(res, out_dir, page_idx=i)

    return len(results)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_ppocr_res_img(res, out_dir: Path, page_idx: int) -> None:
    """Save only the annotated result image (ocr_res_img), skipping
    any preprocessed / intermediate input images."""
    img_dict = res.img  # Dict[str, PIL.Image.Image]
    res_img = img_dict.get("ocr_res_img")
    if res_img is None:
        # Unexpected key layout — fall back to saving all keys
        logger.warning(
            "ocr_res_img not found in result.img keys %s; saving all.", list(img_dict)
        )
        for key, img in img_dict.items():
            img.save(str(out_dir / f"page_{page_idx:04d}_{key}.png"))
        return
    out_path = out_dir / f"page_{page_idx:04d}_res_img.png"
    res_img.save(str(out_path))


def _convert_md_to_html(out_dir: Path) -> None:
    """Convert every *.md file in *out_dir* to a sibling *.html file.

    Uses a minimal pure-stdlib Markdown→HTML converter sufficient for the
    structured output that PaddleOCR-VL produces (headers, paragraphs,
    bold/italic, inline code, fenced code blocks, GFM tables, images,
    horizontal rules).
    """
    for md_path in out_dir.glob("*.md"):
        html_path = md_path.with_suffix(".html")
        html_content = _md_to_html(md_path.read_text(encoding="utf-8"))
        html_path.write_text(html_content, encoding="utf-8")
        logger.debug("  → converted %s → %s", md_path.name, html_path.name)


def _md_to_html(md: str) -> str:
    """Minimal Markdown-to-HTML converter (no external dependencies).

    Handles the subset produced by PaddleOCR-VL:
    - ATX headings (#–######)
    - Fenced code blocks (``` ```)
    - GFM tables (| col | col |)
    - Horizontal rules (---, ***)
    - Images (![alt](src))
    - Bold (**text** / __text__)
    - Italic (*text* / _text_)
    - Inline code (`code`)
    - Paragraphs / line breaks
    """
    lines = md.splitlines()
    html_parts: list[str] = []
    i = 0

    def _inline(text: str) -> str:
        """Apply inline-level transformations."""
        # Images before links
        text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img alt="\1" src="\2">', text)
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        return text

    while i < len(lines):
        line = lines[i]

        # --- Fenced code block ---
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

        # --- ATX heading ---
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            html_parts.append(f"<h{level}>{_inline(heading_match.group(2))}</h{level}>")
            i += 1
            continue

        # --- Horizontal rule ---
        if re.match(r'^(\*{3,}|-{3,}|_{3,})\s*$', line):
            html_parts.append("<hr>")
            i += 1
            continue

        # --- GFM table ---
        if "|" in line and i + 1 < len(lines) and re.match(r'^\|?\s*[-:]+', lines[i + 1]):
            def _parse_row(row: str) -> list[str]:
                return [c.strip() for c in row.strip().strip("|").split("|")]

            headers = _parse_row(line)
            i += 2  # skip separator row
            header_html = "".join(f"<th>{_inline(h)}</th>" for h in headers)
            rows_html: list[str] = []
            while i < len(lines) and "|" in lines[i]:
                cells = _parse_row(lines[i])
                row_html = "".join(f"<td>{_inline(c)}</td>" for c in cells)
                rows_html.append(f"<tr>{row_html}</tr>")
                i += 1
            rows_block = "\n".join(rows_html)
            html_parts.append(
                f"<table>\n<thead><tr>{header_html}</tr></thead>\n"
                f"<tbody>\n{rows_block}\n</tbody>\n</table>"
            )
            continue

        # --- Empty line (paragraph separator) ---
        if line.strip() == "":
            html_parts.append("")
            i += 1
            continue

        # --- Paragraph / plain line ---
        # Collect contiguous non-empty, non-special lines into one <p>
        para_lines: list[str] = []
        while i < len(lines):
            l = lines[i]
            if (
                l.strip() == ""
                or re.match(r'^#{1,6}\s', l)
                or l.strip().startswith("```")
                or re.match(r'^(\*{3,}|-{3,}|_{3,})\s*$', l)
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


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    """Create a zip archive of all files in *src_dir* (flat, no subdirectory prefix)."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(src_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(src_dir))


# ---------------------------------------------------------------------------
# Core batch logic
# ---------------------------------------------------------------------------

def run_batch(
    input_dir: Path,
    output_dir: Path,
    engine_name: str,
    device: str = "gpu:0",
    lang: str = "en",
    merge_tables: bool = True,
    relevel_titles: bool = True,
    log_file: Optional[str] = None,
):
    """Discover PDFs recursively, run inference, and mirror folder structure."""

    # Validate input
    if not input_dir.is_dir():
        logger.error("Input path is not a directory: %s", input_dir)
        sys.exit(1)

    # Collect PDFs
    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found under %s", input_dir)
        return

    logger.info(
        "Found %d PDF file(s) under %s — engine=%s, device=%s",
        len(pdf_files), input_dir, engine_name, device,
    )

    # Build engine once
    if engine_name == "vl":
        engine = _build_vl_engine(device)
    else:
        engine = _build_ppocr_engine(device, lang=lang)

    # Optional tqdm progress bar (graceful fallback)
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
        pdf_iter = tqdm(pdf_files, desc="Processing PDFs", unit="file")
    except ImportError:
        pdf_iter = pdf_files

    total_pages = 0
    success_count = 0
    fail_count = 0
    t_start = time.time()

    for pdf_path in pdf_iter:
        rel = pdf_path.relative_to(input_dir)
        out_dir = output_dir / rel.parent / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing: %s → %s", rel, out_dir)
        t0 = time.time()

        try:
            if engine_name == "vl":
                n_pages = _run_vl(
                    engine, pdf_path, out_dir,
                    merge_tables=merge_tables,
                    relevel_titles=relevel_titles,
                )
            else:
                n_pages = _run_ppocr(engine, pdf_path, out_dir)

            elapsed = time.time() - t0
            total_pages += n_pages
            success_count += 1
            logger.info(
                "  ✓ %s — %d page(s), %.1fs", rel, n_pages, elapsed,
            )
        except Exception:
            fail_count += 1
            logger.exception("  ✗ Failed: %s", rel)

    elapsed_total = time.time() - t_start
    logger.info(
        "Done — %d/%d files succeeded, %d failed, %d total page(s), %.1fs elapsed",
        success_count, len(pdf_files), fail_count, total_pages, elapsed_total,
    )

    # Zip the entire output directory into a single archive next to it.
    zip_path = output_dir.with_suffix(".zip")
    _zip_dir(output_dir, zip_path)
    logger.info("Results zipped → %s", zip_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch PaddleOCR inference on PDF files (PaddleOCR-VL / PP-OCRv5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Root input folder containing PDF files (searched recursively).",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output"),
        help="Root output folder. Directory tree mirrors the input.",
    )
    p.add_argument(
        "--engine", "-e",
        choices=["vl", "ppocr"],
        default="vl",
        help="OCR engine: 'vl' for PaddleOCR-VL 1.5, 'ppocr' for PP-OCRv5.",
    )
    p.add_argument(
        "--device", "-d",
        type=str,
        default="gpu:0",
        help="Compute device, e.g. 'gpu:0', 'cpu'.",
    )
    p.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language for PP-OCRv5 (ignored when engine=vl).",
    )
    p.add_argument(
        "--merge-tables",
        action="store_true",
        default=True,
        help="(VL only) Merge tables spanning multiple pages.",
    )
    p.add_argument(
        "--no-merge-tables",
        dest="merge_tables",
        action="store_false",
        help="(VL only) Disable cross-page table merging.",
    )
    p.add_argument(
        "--relevel-titles",
        action="store_true",
        default=True,
        help="(VL only) Reconstruct multi-level heading hierarchy.",
    )
    p.add_argument(
        "--no-relevel-titles",
        dest="relevel_titles",
        action="store_false",
        help="(VL only) Disable title re-leveling.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write a log file.",
    )

    return p.parse_args(argv)


def _setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure the module logger with console (and optional file) output."""
    global logger
    logger = logging.getLogger("paddleocr_infer")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level))

    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level))
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(getattr(logging, level))
        fh.setFormatter(fmt)
        logger.addHandler(fh)


def main(argv=None):
    args = parse_args(argv)

    # Configure logging from CLI flags
    _setup_logging(level=args.log_level, log_file=args.log_file)

    run_batch(
        input_dir=args.input.resolve(),
        output_dir=args.output.resolve(),
        engine_name=args.engine,
        device=args.device,
        lang=args.lang,
        merge_tables=args.merge_tables,
        relevel_titles=args.relevel_titles,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
