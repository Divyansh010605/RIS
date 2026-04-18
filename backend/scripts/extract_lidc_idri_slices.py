"""Extract PNG slices from an LIDC-IDRI CT dataset for model testing.

Example:
    python extract_lidc_idri_slices.py --input-root "D:/LIDC-IDRI" --output-root "./data/lidc_slices"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH = 1500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CT slice PNGs from LIDC-IDRI DICOM series.")
    parser.add_argument("--input-root", required=True, help="Root folder containing LIDC-IDRI DICOM files.")
    parser.add_argument("--output-root", required=True, help="Folder where PNG slices will be written.")
    parser.add_argument("--stride", type=int, default=1, help="Save every Nth slice from each series.")
    parser.add_argument("--min-slices", type=int, default=5, help="Skip series with fewer than this many slices.")
    parser.add_argument("--window-center", type=int, default=LUNG_WINDOW_CENTER, help="CT window center.")
    parser.add_argument("--window-width", type=int, default=LUNG_WINDOW_WIDTH, help="CT window width.")
    parser.add_argument("--limit-series", type=int, default=0, help="Optional cap on the number of series to process.")
    return parser.parse_args()


def normalize_hounsfield_units(dataset: pydicom.dataset.FileDataset) -> np.ndarray:
    pixels = dataset.pixel_array.astype(np.int16)
    pixels[pixels == -2000] = 0

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    if slope != 1.0:
        pixels = slope * pixels.astype(np.float32)
    pixels = pixels + intercept
    return pixels


def window_image(image: np.ndarray, center: int, width: int) -> np.ndarray:
    minimum = center - (width / 2)
    maximum = center + (width / 2)
    clipped = np.clip(image, minimum, maximum)
    normalized = (clipped - minimum) / (maximum - minimum)
    normalized = (normalized * 255.0).astype(np.uint8)
    return normalized


def sort_key(dataset: pydicom.dataset.FileDataset) -> float:
    position = getattr(dataset, "ImagePositionPatient", None)
    if position and len(position) == 3:
        return float(position[2])
    instance_number = getattr(dataset, "InstanceNumber", None)
    if instance_number is not None:
        return float(instance_number)
    return 0.0


def load_series(dicom_paths: list[Path]) -> list[pydicom.dataset.FileDataset]:
    series = []
    for path in dicom_paths:
        try:
            dataset = pydicom.dcmread(str(path), force=True)
        except Exception:
            continue

        if getattr(dataset, "Modality", None) != "CT":
            continue
        if not hasattr(dataset, "pixel_array"):
            continue
        series.append(dataset)

    series.sort(key=sort_key)
    return series


def extract_series(
    series_uid: str,
    datasets: list[pydicom.dataset.FileDataset],
    output_root: Path,
    stride: int,
    window_center: int,
    window_width: int,
    writer: csv.writer,
) -> int:
    if len(datasets) < 1:
        return 0

    series_output_dir = output_root / series_uid
    series_output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for index, dataset in enumerate(datasets[::stride]):
        pixels = normalize_hounsfield_units(dataset)
        pixels = window_image(pixels, window_center, window_width)
        file_name = f"slice_{index:04d}.png"
        file_path = series_output_dir / file_name
        Image.fromarray(pixels).save(file_path)
        writer.writerow([series_uid, index, str(file_path)])
        saved_count += 1

    return saved_count


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dicom_paths = [path for path in input_root.rglob("*") if path.is_file()]
    series_map: dict[str, list[Path]] = defaultdict(list)

    for path in dicom_paths:
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue

        if getattr(dataset, "Modality", None) != "CT":
            continue

        series_uid = getattr(dataset, "SeriesInstanceUID", None)
        if not series_uid:
            continue
        series_map[series_uid].append(path)

    manifest_path = output_root / "manifest.csv"
    total_saved = 0
    total_series = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["series_uid", "slice_index", "png_path"])

        for series_uid, paths in sorted(series_map.items()):
            if args.limit_series and total_series >= args.limit_series:
                break

            datasets = load_series(paths)
            if len(datasets) < args.min_slices:
                continue

            saved = extract_series(
                series_uid=series_uid,
                datasets=datasets,
                output_root=output_root,
                stride=max(1, args.stride),
                window_center=args.window_center,
                window_width=args.window_width,
                writer=writer,
            )
            if saved > 0:
                total_series += 1
                total_saved += saved
                print(f"Exported {saved} slices from series {series_uid}")

    print(f"Done. Saved {total_saved} PNG slices across {total_series} CT series.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
