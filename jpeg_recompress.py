#!/usr/bin/env python3
"""Python port of jpeg-recompress.

This is a simplified reimplementation that approximates the original C
program. It exposes (most of) the same command line arguments but only
implements a subset of the comparison metrics. The binary search over
JPEG quality is preserved.
"""

import argparse
import math
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


QUALITY_PRESETS = {
    "low": 0.5,
    "medium": 0.75,
    "subhigh": 0.875,
    "high": 0.9375,
    "veryhigh": 0.96875,
}


def metric_mpe(a: np.ndarray, b: np.ndarray) -> float:
    """Mean pixel error."""
    return np.mean(np.abs(a - b))


def metric_mse(a: np.ndarray, b: np.ndarray) -> float:
    return np.mean((a - b) ** 2)


def metric_msef(a: np.ndarray, b: np.ndarray) -> float:
    mse = metric_mse(a, b)
    var = np.var(np.concatenate([a, b]))
    return math.sqrt(mse / (var if var > 0 else 1))


def metric_cor(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def metric_psnr(a: np.ndarray, b: np.ndarray) -> float:
    return peak_signal_noise_ratio(a, b, data_range=1.0)


def metric_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return structural_similarity(a, b, data_range=1.0)


def metric_ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return structural_similarity(a, b, data_range=1.0, multichannel=False,
                                  gaussian_weights=True, sigma=1.5)


METRIC_FUNCS = {
    "mpe": metric_mpe,
    "psnr": metric_psnr,
    "mse": metric_mse,
    "msef": metric_msef,
    "cor": metric_cor,
    "ssim": metric_ssim,
    "ms-ssim": metric_ms_ssim,
}


def compute_metric(method: str, ref: np.ndarray, cmp: np.ndarray) -> float:
    if method not in METRIC_FUNCS:
        raise NotImplementedError(f"Metric '{method}' not implemented")
    return METRIC_FUNCS[method](ref, cmp)


def recompress(in_path: Path, out_path: Path, *, target: float, jpeg_min: int,
               jpeg_max: int, preset: str, loops: int, method: str,
               progressive: bool, accurate: bool):
    with Image.open(in_path) as im:
        im = im.convert("RGB")
        original = np.asarray(im, dtype=np.float32) / 255.0
        gray_ref = np.asarray(im.convert("L"), dtype=np.float32) / 255.0

    if target <= 0 and preset:
        target = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["medium"])

    low = jpeg_min
    high = jpeg_max
    best_q = high

    for attempt in range(loops):
        q = (low + high + 1) // 2
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=False,
                 progressive=False)
        buf.seek(0)
        cmp_im = Image.open(buf).convert("L")
        gray_cmp = np.asarray(cmp_im, dtype=np.float32) / 255.0

        metric = compute_metric(method, gray_ref, gray_cmp)
        if metric >= target:
            best_q = q
            high = q - 1
        else:
            low = q + 1

    im.save(out_path, format="JPEG", quality=best_q,
             optimize=accurate, progressive=progressive)


def main():
    parser = argparse.ArgumentParser(description="Python port of jpeg-recompress")
    parser.add_argument("input", help="input image path")
    parser.add_argument("output", help="output image path")

    parser.add_argument("-a", "--accurate", action="store_true", help="favor accuracy over speed")
    parser.add_argument("-c", "--no-copy", action="store_true", help="disable copying files that will not be compressed")
    parser.add_argument("-d", "--defish", type=float, default=0.0, metavar="STRENGTH", help="set defish strength")
    parser.add_argument("-f", "--force", action="store_true", help="force process")
    parser.add_argument("-l", "--loops", type=int, default=6, help="set number of runs to attempt")
    parser.add_argument("-m", "--method", default="sum", help="comparison method")
    parser.add_argument("-n", "--min", dest="jpeg_min", type=int, default=40, help="minimum JPEG quality")
    parser.add_argument("-p", "--no-progressive", action="store_true", help="disable progressive encoding")
    parser.add_argument("-q", "--quality", choices=list(QUALITY_PRESETS.keys()), default="medium", help="quality preset")
    parser.add_argument("-r", "--ppm", action="store_true", help="parse input as PPM")
    parser.add_argument("-s", "--strip", action="store_true", help="strip metadata")
    parser.add_argument("-t", "--target", type=float, default=0.0, help="target quality")
    parser.add_argument("-x", "--max", dest="jpeg_max", type=int, default=98, help="maximum JPEG quality")
    parser.add_argument("-z", "--zoom", type=float, default=1.0, help="defish zoom")
    parser.add_argument("-Q", "--quiet", action="store_true", help="only print out errors")
    parser.add_argument("-S", "--subsample", default="default", help="set subsampling method")
    parser.add_argument("-T", "--input-filetype", default="auto", help="set input file type")
    parser.add_argument("-V", "--version", action="version", version="python-port")
    parser.add_argument("-Y", "--ycbcr", type=int, default=0, help="YCbCr jpeg colorspace")

    args = parser.parse_args()
    recompress(
        Path(args.input),
        Path(args.output),
        target=args.target,
        jpeg_min=args.jpeg_min,
        jpeg_max=args.jpeg_max,
        preset=args.quality,
        loops=args.loops,
        method=args.method,
        progressive=not args.no_progressive,
        accurate=args.accurate,
    )


if __name__ == "__main__":
    main()
