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
import sys
import time
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
    """Run PaddleOCR-VL inference on a single PDF and save native outputs."""
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

    return len(results)


def _run_ppocr(engine, pdf_path: Path, out_dir: Path):
    """Run PP-OCRv5 inference on a single PDF and save native outputs."""
    results = list(engine.predict(input=str(pdf_path)))

    for res in results:
        res.save_to_json(save_path=str(out_dir))
        res.save_to_img(save_path=str(out_dir))

    return len(results)


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
