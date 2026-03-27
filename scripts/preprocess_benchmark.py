"""
Standalone benchmark: does image preprocessing help grounding models find
low-contrast UI widgets?

Usage
-----
  conda run -n omni python scripts/preprocess_benchmark.py \
      --image screenshot.png \
      --target "username input field"

The script will open an interactive window so you can click the target widget
to mark ground truth, then query both OmniParser and GTA1 for every
preprocessing variant and report results.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import sys
from typing import Any

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

MAX_WIDTH = 1280  # match the pipeline's resize ceiling


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def resize_to_max_width(img: Image.Image, max_width: int = MAX_WIDTH) -> Image.Image:
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    return img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------

def variant_raw(img: Image.Image) -> Image.Image:
    return img.copy()


def variant_clahe(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def variant_adaptive_thresh(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(arr, 0.6, thresh_rgb, 0.4, 0)
    return Image.fromarray(blended)


def variant_edge_overlay(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_rgb = np.zeros_like(arr)
    edge_rgb[:, :, 0] = edges  # red channel
    blended = cv2.addWeighted(arr, 0.7, edge_rgb, 0.3, 0)
    return Image.fromarray(blended)


def variant_clahe_edges(img: Image.Image) -> Image.Image:
    return variant_edge_overlay(variant_clahe(img))


VARIANTS: list[tuple[str, Any]] = [
    ("raw", variant_raw),
    ("clahe", variant_clahe),
    ("adaptive_thresh", variant_adaptive_thresh),
    ("edge_overlay", variant_edge_overlay),
    ("clahe+edges", variant_clahe_edges),
]


# ---------------------------------------------------------------------------
# Ground truth collection
# ---------------------------------------------------------------------------

def collect_ground_truth(img: Image.Image) -> tuple[int, int]:
    """Open an interactive window; return (x, y) of the user's click."""
    print("\n[GT] Click on the target widget in the window that opens.")
    print("     Close the window after clicking (or press any key).\n")

    fig, ax = plt.subplots(figsize=(min(img.width / 96, 16), min(img.height / 96, 12)))
    ax.imshow(img)
    ax.set_title("Click the target widget, then close this window", fontsize=11)
    ax.axis("off")

    clicked: list[tuple[int, int]] = []

    def on_click(event):
        if event.inaxes is ax and event.button == 1:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            clicked.append((x, y))
            ax.plot(x, y, "go", markersize=12)
            ax.set_title(f"Ground truth: ({x}, {y}) — close window to continue", fontsize=11)
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if not clicked:
        print("[GT] No click registered. Exiting.")
        sys.exit(1)

    gt_x, gt_y = clicked[-1]
    print(f"[GT] Ground truth set to ({gt_x}, {gt_y})")
    return gt_x, gt_y


# ---------------------------------------------------------------------------
# Grounding backends
# ---------------------------------------------------------------------------

def parse_omniparser(b64: str, url: str, timeout: int = 120) -> dict:
    resp = requests.post(f"{url}/parse", json={"base64_image": b64}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ground_gta1(b64: str, instruction: str, url: str, timeout: int = 30) -> tuple[int, int]:
    resp = requests.post(
        f"{url}/ground",
        json={"image_base64": b64, "instruction": instruction},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return int(data["x"]), int(data["y"])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def nearest_element_from_omniparser(
    elements: list[dict], gt_x: int, gt_y: int, img_w: int, img_h: int
) -> tuple[dict | None, float]:
    """Return the element whose centroid is closest to (gt_x, gt_y) and its pixel distance."""
    best_elem, best_dist = None, float("inf")
    for elem in elements:
        bbox = elem.get("bbox", [])
        if len(bbox) < 4:
            continue
        cx = int((bbox[0] + bbox[2]) / 2 * img_w)
        cy = int((bbox[1] + bbox[3]) / 2 * img_h)
        dist = math.hypot(cx - gt_x, cy - gt_y)
        if dist < best_dist:
            best_dist = dist
            best_elem = {**elem, "_cx": cx, "_cy": cy}
    return best_elem, best_dist


def is_hit(dist: float, tolerance: int) -> bool:
    return dist <= tolerance


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

CIRCLE_RADIUS = 14
CROSS_SIZE = 14
CROSS_WIDTH = 3


def annotate_pil(img: Image.Image, gt: tuple[int, int], pred: tuple[int, int] | None) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    gx, gy = gt
    # Green circle = ground truth
    draw.ellipse(
        [gx - CIRCLE_RADIUS, gy - CIRCLE_RADIUS, gx + CIRCLE_RADIUS, gy + CIRCLE_RADIUS],
        outline=(0, 220, 0),
        width=3,
    )
    # Red cross = prediction
    if pred is not None:
        px, py = pred
        draw.line(
            [px - CROSS_SIZE, py, px + CROSS_SIZE, py], fill=(220, 0, 0), width=CROSS_WIDTH
        )
        draw.line(
            [px, py - CROSS_SIZE, px, py + CROSS_SIZE], fill=(220, 0, 0), width=CROSS_WIDTH
        )
    return out


def build_result_grid(
    results: list[dict],
    output_path: str,
    gt: tuple[int, int],
    img_w: int,
    img_h: int,
) -> None:
    """
    Grid layout: rows = variants, cols = (preprocessed | OmniParser SOM | GTA1 annotated)
    """
    n_rows = len(results)
    n_cols = 3
    dpi = 96
    cell_w = min(img_w, MAX_WIDTH)
    cell_h = int(cell_w * img_h / img_w)
    fig_w = n_cols * cell_w / dpi
    fig_h = n_rows * cell_h / dpi + 0.5  # extra space for col headers
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
    if n_rows == 1:
        axes = [axes]

    col_titles = ["Preprocessed input", "OmniParser SOM", "GTA1 prediction"]

    for col_idx, title in enumerate(col_titles):
        axes[0][col_idx].set_title(title, fontsize=9, fontweight="bold")

    for row_idx, res in enumerate(results):
        variant_name = res["variant"]
        row_axes = axes[row_idx]

        # Col 0: preprocessed image with GT circle
        pre_img = res["preprocessed_img"]
        pre_ann = annotate_pil(pre_img, gt, None)
        row_axes[0].imshow(pre_ann)
        row_axes[0].set_ylabel(variant_name, fontsize=8, rotation=0, labelpad=60, va="center")
        row_axes[0].axis("off")

        # Col 1: OmniParser SOM output annotated with GT + nearest element
        omni_result = res.get("omniparser")
        if omni_result and "som_img" in omni_result:
            som_img = omni_result["som_img"]
            pred = omni_result.get("pred_xy")
            ann = annotate_pil(som_img, gt, pred)
            hit = omni_result.get("hit", False)
            dist = omni_result.get("dist", float("inf"))
            status = f"HIT {dist:.0f}px" if hit else f"MISS {dist:.0f}px"
            row_axes[1].imshow(ann)
            row_axes[1].set_xlabel(status, fontsize=8, color="green" if hit else "red")
        else:
            err = omni_result.get("error", "N/A") if omni_result else "N/A"
            row_axes[1].text(0.5, 0.5, f"Error:\n{err}", ha="center", va="center", fontsize=7, wrap=True)
            row_axes[1].set_facecolor("#ffe0e0")
        row_axes[1].axis("off")

        # Col 2: raw image annotated with GT + GTA1 prediction
        gta_result = res.get("gta1")
        if gta_result and "pred_xy" in gta_result:
            pred = gta_result["pred_xy"]
            ann = annotate_pil(res["preprocessed_img"], gt, pred)
            hit = gta_result.get("hit", False)
            dist = gta_result.get("dist", float("inf"))
            status = f"HIT {dist:.0f}px" if hit else f"MISS {dist:.0f}px"
            row_axes[2].imshow(ann)
            row_axes[2].set_xlabel(status, fontsize=8, color="green" if hit else "red")
        else:
            err = gta_result.get("error", "N/A") if gta_result else "N/A"
            row_axes[2].text(0.5, 0.5, f"Error:\n{err}", ha="center", va="center", fontsize=7, wrap=True)
            row_axes[2].set_facecolor("#ffe0e0")
        row_axes[2].axis("off")

    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="green", linewidth=2, label="Ground truth (GT)"),
        mpatches.Patch(facecolor="none", edgecolor="red", linewidth=2, label="Model prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8, frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")
    print(f"\n[VIZ] Saved result grid to: {output_path}")


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    header = f"{'Variant':<18} | {'OmniParser':^22} | {'GTA1':^22}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for res in results:
        variant = res["variant"]

        omni = res.get("omniparser")
        if omni and "dist" in omni:
            hit = "HIT " if omni["hit"] else "MISS"
            omni_str = f"{hit}  dist={omni['dist']:5.1f}px"
        else:
            omni_str = f"{'ERROR':^22}"

        gta = res.get("gta1")
        if gta and "dist" in gta:
            hit = "HIT " if gta["hit"] else "MISS"
            gta_str = f"{hit}  dist={gta['dist']:5.1f}px"
        else:
            gta_str = f"{'ERROR':^22}"

        print(f"{variant:<18} | {omni_str:^22} | {gta_str:^22}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark image preprocessing variants for grounding accuracy."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to screenshot PNG/JPEG")
    src.add_argument("--capture", action="store_true", help="Capture live screen (requires pyautogui)")
    parser.add_argument("--target", required=True, help='Natural-language target, e.g. "username input field"')
    parser.add_argument("--omniparser-url", default="http://localhost:8000")
    parser.add_argument("--gta1-url", default="http://localhost:8002")
    parser.add_argument("--tolerance", type=int, default=30, help="Hit radius in pixels (default: 30)")
    parser.add_argument("--output", default="preprocessing_results.png")
    parser.add_argument("--skip-omniparser", action="store_true")
    parser.add_argument("--skip-gta1", action="store_true")
    args = parser.parse_args()

    # --- Load image ---
    if args.capture:
        try:
            import pyautogui
            pil_img = pyautogui.screenshot()
            print("[IMG] Captured live screenshot.")
        except ImportError:
            print("[ERROR] pyautogui not installed. Use --image instead.")
            sys.exit(1)
    else:
        pil_img = Image.open(args.image).convert("RGB")
        print(f"[IMG] Loaded {args.image} ({pil_img.width}x{pil_img.height})")

    orig_w, orig_h = pil_img.size
    pil_img = resize_to_max_width(pil_img)
    w, h = pil_img.size
    if (orig_w, orig_h) != (w, h):
        print(f"[IMG] Resized to {w}x{h}")

    # --- Collect ground truth ---
    gt_x, gt_y = collect_ground_truth(pil_img)

    # --- Run benchmark ---
    results = []
    for variant_name, fn in VARIANTS:
        print(f"\n[RUN] Variant: {variant_name}")
        processed = fn(pil_img)
        b64 = pil_to_b64(processed)
        entry: dict = {"variant": variant_name, "preprocessed_img": processed}

        # OmniParser
        if not args.skip_omniparser:
            try:
                resp = parse_omniparser(b64, args.omniparser_url)
                som_b64 = resp.get("labeled_screenshot_base64", b64)
                elements = resp.get("parsed_content_list", [])
                som_img = b64_to_pil(som_b64) if som_b64 else processed
                nearest, dist = nearest_element_from_omniparser(elements, gt_x, gt_y, w, h)
                pred_xy = (nearest["_cx"], nearest["_cy"]) if nearest else None
                hit = is_hit(dist, args.tolerance)
                print(f"  OmniParser: {len(elements)} elements detected, "
                      f"nearest dist={dist:.1f}px → {'HIT' if hit else 'MISS'}")
                entry["omniparser"] = {
                    "som_img": som_img,
                    "pred_xy": pred_xy,
                    "dist": dist,
                    "hit": hit,
                    "n_elements": len(elements),
                }
            except Exception as e:
                print(f"  OmniParser: ERROR — {e}")
                entry["omniparser"] = {"error": str(e)}
        else:
            entry["omniparser"] = {"error": "skipped"}

        # GTA1
        if not args.skip_gta1:
            try:
                gx, gy = ground_gta1(b64, args.target, args.gta1_url)
                dist = math.hypot(gx - gt_x, gy - gt_y)
                hit = is_hit(dist, args.tolerance)
                print(f"  GTA1: predicted ({gx}, {gy}), dist={dist:.1f}px → {'HIT' if hit else 'MISS'}")
                entry["gta1"] = {"pred_xy": (gx, gy), "dist": dist, "hit": hit}
            except Exception as e:
                print(f"  GTA1: ERROR — {e}")
                entry["gta1"] = {"error": str(e)}
        else:
            entry["gta1"] = {"error": "skipped"}

        results.append(entry)

    # --- Report ---
    print_table(results)
    build_result_grid(results, args.output, (gt_x, gt_y), w, h)


if __name__ == "__main__":
    # Use a non-interactive backend if running headless; override with TkAgg for interactive GT collection
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    main()
