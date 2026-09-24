"""
Amazon ML Worker — Synthetic Data Generator.

Creates synthetic tabular data with text, numeric, categorical columns, and target.
Optionally creates small synthetic images.
Sufficient to exercise the full worker pipeline before real data arrives.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.utils.logging import get_logger

log = get_logger(__name__)

# Sample text corpus for generating synthetic text
_SAMPLE_TEXTS = [
    "Premium quality product with excellent build",
    "Compact design suitable for everyday use",
    "Ergonomic handle with non-slip grip material",
    "Lightweight aluminum frame, weighs only 500g",
    "Available in red, blue, green, black, and white",
    "Waterproof rating IPX7, suitable for outdoor use",
    "Battery life up to 12 hours on single charge",
    "Compatible with USB-C fast charging",
    "Dimensions: 15cm x 8cm x 3cm",
    "Made from recycled materials, eco-friendly",
    "Noise cancellation technology for clear audio",
    "Multi-functional device with 5 operating modes",
    "Temperature range: -20°C to 60°C",
    "High resolution display with 300 DPI",
    "Stainless steel construction, rust resistant",
    "Capacity: 2 litres, ideal for travel",
    "Energy efficient, rated A++ for low consumption",
    "Foldable design for easy storage and transport",
    "Comes with 2-year manufacturer warranty",
    "Package includes: main unit, cable, and manual",
]

_CATEGORIES = ["electronics", "clothing", "home", "sports", "books", "toys", "food", "beauty"]
_BRANDS = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "BrandF"]
_COLORS = ["red", "blue", "green", "black", "white", "silver", "gold"]


def generate_synthetic_dataset(
    n_rows: int = 1000,
    n_classes: int = 5,
    output_dir: str = "synthetic_data",
    create_images: bool = True,
    image_count: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset for pipeline testing.

    Creates:
      - synthetic_data/train.csv
      - synthetic_data/test.csv
      - synthetic_data/images/ (optional)
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate IDs
    ids = [f"SYNTH_{i:06d}" for i in range(n_rows)]

    # Generate features
    data = {
        "id": ids,
        "text": [random.choice(_SAMPLE_TEXTS) + " " + " ".join(
            random.choices(_SAMPLE_TEXTS, k=rng.randint(0, 3))
        ) for _ in range(n_rows)],
        "category": rng.choice(_CATEGORIES, n_rows).tolist(),
        "brand": rng.choice(_BRANDS, n_rows).tolist(),
        "color": rng.choice(_COLORS, n_rows).tolist(),
        "price": np.round(rng.uniform(5.0, 500.0, n_rows), 2).tolist(),
        "weight_kg": np.round(rng.uniform(0.1, 50.0, n_rows), 2).tolist(),
        "rating": np.round(rng.uniform(1.0, 5.0, n_rows), 1).tolist(),
        "num_reviews": rng.randint(0, 10000, n_rows).tolist(),
        "target": rng.randint(0, n_classes, n_rows).tolist(),
    }

    # Add some missing values
    n_missing = max(1, n_rows // 20)
    missing_idx = rng.choice(n_rows, n_missing, replace=False)
    for idx in missing_idx:
        data["text"][idx] = None
    missing_idx2 = rng.choice(n_rows, n_missing, replace=False)
    for idx in missing_idx2:
        data["weight_kg"][idx] = None

    df = pd.DataFrame(data)

    # Create images
    if create_images:
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        actual_count = min(image_count, n_rows)
        image_refs = []
        for i in range(n_rows):
            if i < actual_count:
                img_name = f"img_{i:06d}.png"
                _create_synthetic_image(str(img_dir / img_name), rng)
                image_refs.append(img_name)
            else:
                # Reference an existing image (reuse)
                ref_idx = i % actual_count
                image_refs.append(f"img_{ref_idx:06d}.png")
        df["image"] = image_refs

    # Split into train/test
    split_idx = int(n_rows * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # Test set: remove target
    test_df_no_target = test_df.drop(columns=["target"])

    # Save
    train_path = out / "train.csv"
    test_path = out / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df_no_target.to_csv(test_path, index=False)

    # Also save as parquet
    train_df.to_parquet(out / "train.parquet", index=False)

    # Save ground truth for test (for evaluation)
    test_df[["id", "target"]].to_csv(out / "test_labels.csv", index=False)

    log.info(
        "Generated synthetic data: %d train, %d test (%d classes) -> %s",
        len(train_df), len(test_df_no_target), n_classes, out,
    )

    return train_df


def _create_synthetic_image(path: str, rng: np.random.RandomState) -> None:
    """Create a small synthetic image with random shapes."""
    width, height = 64, 64
    img = Image.new("RGB", (width, height), color=(
        rng.randint(200, 255),
        rng.randint(200, 255),
        rng.randint(200, 255),
    ))
    draw = ImageDraw.Draw(img)

    # Draw random shapes
    n_shapes = rng.randint(1, 5)
    for _ in range(n_shapes):
        color = (rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 200))
        x1, y1 = rng.randint(0, width // 2), rng.randint(0, height // 2)
        x2, y2 = rng.randint(width // 2, width), rng.randint(height // 2, height)
        shape_type = rng.choice(["rect", "ellipse"])
        if shape_type == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)

    img.save(path)


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging()
    generate_synthetic_dataset()
