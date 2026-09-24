"""Generate synthetic data for pipeline testing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from synthetic_data.generate import generate_synthetic_dataset

def main():
    setup_logging()
    generate_synthetic_dataset(
        n_rows=1000,
        n_classes=5,
        output_dir="synthetic_data",
        create_images=True,
        image_count=50,
        seed=42,
    )
    print("\nSynthetic data generated in synthetic_data/")
    print("  - train.csv / train.parquet")
    print("  - test.csv")
    print("  - test_labels.csv")
    print("  - images/ (50 synthetic images)")

if __name__ == "__main__":
    main()
