#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from suspicious_cases import central_ellipse_mask, border_mask, normalize_float01


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


def load_grayscale_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image)


def detect_inversion_features(
    image: np.ndarray,
    border_frac: float = 0.08,
) -> dict[str, float]:

    img = normalize_float01(image)
    h, w = img.shape

    border = border_mask(h, w, border_frac=border_frac).astype(bool)
    center = central_ellipse_mask(h, w).astype(bool)

    border_mean = float(img[border].mean())
    center_mean = float(img[center].mean())

    border_median = float(np.median(img[border]))
    center_median = float(np.median(img[center]))

    border_minus_center_mean = border_mean - center_mean
    border_minus_center_median = border_median - center_median

    center_minus_border_mean = center_mean - border_mean

    return {
        "border_mean": border_mean,
        "center_mean": center_mean,
        "border_median": border_median,
        "center_median": center_median,
        "border_minus_center_mean": border_minus_center_mean,
        "border_minus_center_median": border_minus_center_median,
        "center_minus_border_mean": center_minus_border_mean,
    }


def make_debug_montage(
    df: pd.DataFrame,
    output_path: Path,
    max_images: int = 40,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flagged = df[df["likely_inverted"] == 1].copy()
    flagged = flagged.sort_values("inversion_score", ascending=False).head(max_images)

    if flagged.empty:
        print("[INFO] No flagged images for debug montage.")
        return

    n = len(flagged)
    n_cols = 8
    n_rows = int(np.ceil(n / n_cols))

    plt.figure(figsize=(2.2 * n_cols, 2.5 * n_rows))

    for i, (_, row) in enumerate(flagged.iterrows(), start=1):
        image_path = PROJECT_ROOT / row["image_path"]
        image = load_grayscale_image(image_path)

        plt.subplot(n_rows, n_cols, i)
        plt.imshow(image, cmap="gray")
        plt.title(
            f"{row['split']}, KL {row['label']}\nscore={row['inversion_score']:.3f}",
            fontsize=8,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect likely inverted/problematic knee X-ray images using "
            "border-to-center intensity gradient."
        )
    )

    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        type=Path,
        help="One or more split CSVs, e.g. data/splits/train.csv data/splits/val.csv data/splits/test.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/data_quality",
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help=(
            "Flag image as likely inverted if border_minus_center_mean >= threshold. "
            "Used unless --top-n is provided."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=(
            "Optional: flag exactly the top N images by border_minus_center_mean. "
            "Use this if visual inspection suggests approximately N problematic cases."
        ),
    )

    parser.add_argument(
        "--border-frac",
        type=float,
        default=0.08,
        help="Border width fraction used for border intensity measurement.",
    )

    parser.add_argument(
        "--max-debug-images",
        type=int,
        default=40,
        help="Number of flagged examples to show in debug montage.",
    )

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for csv_path_arg in args.csv:
        csv_path = PROJECT_ROOT / csv_path_arg

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        split_name = csv_path.stem
        split_df = pd.read_csv(csv_path)

        if "image_path" not in split_df.columns:
            raise ValueError(f"{csv_path} must contain image_path column.")

        if "label" not in split_df.columns:
            raise ValueError(f"{csv_path} must contain label column.")

        for idx, row in split_df.iterrows():
            rel_image_path = str(row["image_path"])
            image_path = PROJECT_ROOT / rel_image_path

            if not image_path.exists():
                all_rows.append({
                    "split": split_name,
                    "image_path": rel_image_path,
                    "label": row["label"],
                    "missing_image": 1,
                    "likely_inverted": 0,
                    "inversion_score": np.nan,
                })
                continue

            image = load_grayscale_image(image_path)
            features = detect_inversion_features(
                image=image,
                border_frac=args.border_frac,
            )

            inversion_score = features["border_minus_center_mean"]

            all_rows.append({
                "split": split_name,
                "image_path": rel_image_path,
                "label": int(row["label"]),
                "missing_image": 0,
                "inversion_score": float(inversion_score),
                **features,
            })

    qc_df = pd.DataFrame(all_rows)

    if args.top_n is not None:
        qc_df["likely_inverted"] = 0

        valid = qc_df[qc_df["missing_image"] == 0].copy()
        top_paths = (
            valid
            .sort_values("inversion_score", ascending=False)
            .head(args.top_n)["image_path"]
            .tolist()
        )

        qc_df.loc[qc_df["image_path"].isin(top_paths), "likely_inverted"] = 1
        flagging_rule = f"top_n={args.top_n}"
    else:
        qc_df["likely_inverted"] = (
            (qc_df["missing_image"] == 0)
            & (qc_df["inversion_score"] >= args.threshold)
        ).astype(int)
        flagging_rule = f"threshold={args.threshold}"

    qc_path = output_dir / "inversion_qc.csv"
    qc_df.to_csv(qc_path, index=False)

    summary_rows = [
        {"metric": "num_images", "value": int(len(qc_df))},
        {"metric": "num_missing_images", "value": int(qc_df["missing_image"].sum())},
        {"metric": "num_likely_inverted", "value": int(qc_df["likely_inverted"].sum())},
        {
            "metric": "likely_inverted_fraction",
            "value": float(qc_df["likely_inverted"].mean()),
        },
        {"metric": "flagging_rule", "value": flagging_rule},
        {"metric": "border_frac", "value": args.border_frac},
        {
            "metric": "mean_inversion_score",
            "value": float(qc_df["inversion_score"].mean()),
        },
        {
            "metric": "median_inversion_score",
            "value": float(qc_df["inversion_score"].median()),
        },
        {
            "metric": "p95_inversion_score",
            "value": float(qc_df["inversion_score"].quantile(0.95)),
        },
        {
            "metric": "p99_inversion_score",
            "value": float(qc_df["inversion_score"].quantile(0.99)),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "inversion_qc_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Class
    breakdown = (
        qc_df
        .groupby(["split", "label"], dropna=False)
        .agg(
            n=("image_path", "count"),
            num_likely_inverted=("likely_inverted", "sum"),
            mean_inversion_score=("inversion_score", "mean"),
            median_inversion_score=("inversion_score", "median"),
        )
        .reset_index()
    )

    breakdown["likely_inverted_fraction"] = (
        breakdown["num_likely_inverted"] / breakdown["n"].clip(lower=1)
    )

    breakdown_path = output_dir / "inversion_qc_by_split_label.csv"
    breakdown.to_csv(breakdown_path, index=False)

    debug_path = output_dir / "inversion_debug" / "flagged_examples.png"
    make_debug_montage(
        df=qc_df,
        output_path=debug_path,
        max_images=args.max_debug_images,
    )

    print()
    print("Saved inversion QC outputs:")
    print(f"  QC CSV:          {qc_path}")
    print(f"  Summary:         {summary_path}")
    print(f"  Split/label CSV: {breakdown_path}")
    print(f"  Debug montage:   {debug_path}")

    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

