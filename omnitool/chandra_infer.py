"""
Chandra batch inference CLI for PDFs.

Recursively processes all PDF files in an input folder using Chandra
(a layout-aware OCR VLM based on Qwen3-VL), writing Markdown, HTML,
JSON chunks, and extracted images into an output folder that mirrors
the input directory tree.

Requires:
    pip install chandra-ocr

Usage:
    python -m omnitool.chandra_infer --input ./docs --output ./results
    python -m omnitool.chandra_infer --input ./docs --method hf --batch-size 4
    python -m omnitool.chandra_infer --input ./docs --method vllm --max-workers 32
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("chandra_infer")


# ---------------------------------------------------------------------------
# Engine wrapper
# ---------------------------------------------------------------------------

def _build_engine(method: str):
    """Instantiate Chandra InferenceManager (lazy import)."""
    from chandra.model import InferenceManager  # type: ignore[import-untyped]

    engine = InferenceManager(method=method)
    logger.info("Initialized Chandra engine (method=%s)", method)
    return engine


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def _run_chandra(
    engine,
    pdf_path: Path,
    out_dir: Path,
    *,
    batch_size: int,
    max_output_tokens: int,
    include_images: bool,
    include_headers_footers: bool,
    save_html: bool,
    method: str,
    max_workers: int,
    max_retries: int,
) -> int:
    """Run Chandra inference on a single PDF and save native outputs.

    Returns the number of pages processed.
    """
    from chandra.input import load_pdf_images  # type: ignore[import-untyped]
    from chandra.model.schema import BatchInputItem  # type: ignore[import-untyped]

    # Load all pages from PDF as PIL images
    images = load_pdf_images(str(pdf_path), page_range=None)
    if not images:
        logger.warning("No pages extracted from %s", pdf_path)
        return 0

    n_pages = len(images)

    # Build batch items
    batch = [
        BatchInputItem(image=img, prompt_type="ocr_layout")
        for img in images
    ]

    # Process in batches
    all_results = []
    for start in range(0, len(batch), batch_size):
        chunk = batch[start : start + batch_size]
        gen_kwargs = dict(
            max_output_tokens=max_output_tokens,
            include_images=include_images,
            include_headers_footers=include_headers_footers,
        )
        if method == "vllm":
            gen_kwargs["max_workers"] = max_workers
            gen_kwargs["max_retries"] = max_retries
        results = engine.generate(chunk, **gen_kwargs)
        all_results.extend(results)

    # --- Save outputs ---
    stem = pdf_path.stem

    # Concatenated Markdown (all pages)
    md_parts = []
    for res in all_results:
        md_parts.append(res.markdown)

    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n\n---\n\n".join(md_parts), encoding="utf-8")

    # HTML (optional)
    if save_html:
        html_parts = []
        for res in all_results:
            html_parts.append(res.html)
        html_path = out_dir / f"{stem}.html"
        html_path.write_text("\n".join(html_parts), encoding="utf-8")

    # JSON chunks (layout blocks with bounding boxes)
    all_chunks = []
    for i, res in enumerate(all_results):
        page_data = {
            "page": i,
            "page_box": res.page_box,
            "token_count": res.token_count,
            "error": res.error,
            "chunks": res.chunks,
        }
        all_chunks.append(page_data)

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Extracted images
    if include_images:
        img_dir = out_dir / "images"
        for res in all_results:
            if res.images:
                img_dir.mkdir(parents=True, exist_ok=True)
                for fname, pil_img in res.images.items():
                    pil_img.save(str(img_dir / fname))

    return n_pages


# ---------------------------------------------------------------------------
# Core batch logic
# ---------------------------------------------------------------------------

def run_batch(
    input_dir: Path,
    output_dir: Path,
    method: str = "hf",
    batch_size: int = 1,
    max_output_tokens: int = 12384,
    include_images: bool = True,
    include_headers_footers: bool = False,
    save_html: bool = True,
    max_workers: int = 64,
    max_retries: int = 6,
):
    """Discover PDFs recursively, run Chandra inference, mirror folder structure."""

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
        "Found %d PDF file(s) under %s — method=%s, batch_size=%d",
        len(pdf_files), input_dir, method, batch_size,
    )

    # Build engine once
    engine = _build_engine(method)

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
            n_pages = _run_chandra(
                engine, pdf_path, out_dir,
                batch_size=batch_size,
                max_output_tokens=max_output_tokens,
                include_images=include_images,
                include_headers_footers=include_headers_footers,
                save_html=save_html,
                method=method,
                max_workers=max_workers,
                max_retries=max_retries,
            )
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch Chandra OCR inference on PDF files.",
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
        "--method", "-m",
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend: 'hf' for HuggingFace local, 'vllm' for vLLM server.",
    )
    p.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Pages per inference batch (increase for vllm).",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=12384,
        help="Maximum output tokens per page.",
    )
    p.add_argument(
        "--include-images",
        dest="include_images",
        action="store_true",
        default=True,
        help="Extract and save embedded images.",
    )
    p.add_argument(
        "--no-images",
        dest="include_images",
        action="store_false",
        help="Skip image extraction.",
    )
    p.add_argument(
        "--include-headers-footers",
        dest="include_headers_footers",
        action="store_true",
        default=False,
        help="Include page headers and footers in output.",
    )
    p.add_argument(
        "--save-html",
        dest="save_html",
        action="store_true",
        default=True,
        help="Save HTML output with bounding boxes.",
    )
    p.add_argument(
        "--no-html",
        dest="save_html",
        action="store_false",
        help="Skip HTML output.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="(vllm only) Number of parallel workers.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="(vllm only) Max retries per request.",
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
    logger = logging.getLogger("chandra_infer")
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
        method=args.method,
        batch_size=args.batch_size,
        max_output_tokens=args.max_output_tokens,
        include_images=args.include_images,
        include_headers_footers=args.include_headers_footers,
        save_html=args.save_html,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
