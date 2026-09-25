"""
Generate synthetic entity resolution datasets for testing.

Creates realistic multi-source business entity data with ground truth matches.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Business name templates
BUSINESS_NAMES = [
    "Acme Corporation", "Global Tech Solutions", "Smith & Partners Law Firm",
    "Golden Dragon Restaurant", "Pacific Coast Electronics", "Mountain View Dental",
    "Sunrise Bakery", "Blue Ocean Logistics", "Premier Auto Services",
    "Central Park Medical Center", "Green Valley Farms", "Diamond Software Inc",
    "Silver Star Hotels", "Royal Palace Restaurant", "Eagle Eye Security",
    "Metro Construction Co", "City Lights Photography", "Fresh Start Cleaning",
    "Thunder Bay Fishing", "Harbor View Marina", "Alpine Sports Equipment",
    "Redwood Financial Group", "Bright Future Academy", "Crystal Clear Windows",
    "Iron Horse Brewery", "Meadow Lane Florist", "Summit Tech Consulting",
    "Riverstone Properties", "Coastal Breeze Travel", "Heritage Antiques",
    "Maple Leaf Consulting", "Bistrot Parisien", "Boulangerie Delacroix",
    "Chateau Fontaine", "Atelier du Vin", "Taj Mahal Spices",
    "Mumbai Digital Solutions", "Delhi Express Logistics", "Chennai Auto Works",
    "Bangalore Software Labs",
]

ADDRESSES_US = [
    "123 Main Street, Suite 100, New York, NY 10001",
    "456 Oak Avenue, Los Angeles, CA 90001",
    "789 Pine Road, Chicago, IL 60601",
    "321 Elm Street, Houston, TX 77001",
    "654 Cedar Lane, Phoenix, AZ 85001",
    "987 Maple Drive, San Francisco, CA 94101",
    "147 Birch Court, Seattle, WA 98101",
    "258 Walnut Way, Denver, CO 80201",
    "369 Cherry Blvd, Miami, FL 33101",
    "741 Spruce Circle, Boston, MA 02101",
]

ADDRESSES_INDIA = [
    "42 MG Road, Bangalore, Karnataka 560001",
    "15 Nehru Place, New Delhi 110019",
    "78 Park Street, Kolkata, West Bengal 700016",
    "3 Marine Drive, Mumbai, Maharashtra 400020",
    "201 Anna Salai, Chennai, Tamil Nadu 600002",
    "56 Jubilee Hills, Hyderabad, Telangana 500033",
    "88 Civil Lines, Jaipur, Rajasthan 302006",
    "12 Mall Road, Shimla, Himachal Pradesh 171001",
]

ADDRESSES_FRANCE = [
    "15 Rue de Rivoli, 75001 Paris",
    "28 Avenue des Champs-Elysees, 75008 Paris",
    "42 Boulevard Saint-Germain, 75005 Paris",
    "7 Place Bellecour, 69002 Lyon",
    "33 Rue de la Republique, 13001 Marseille",
    "19 Quai des Chartrons, 33000 Bordeaux",
    "5 Rue du Vieux Port, 06300 Nice",
    "11 Place du Capitole, 31000 Toulouse",
]

COUNTRIES = ["US", "India", "France"]


def _add_noise(name: str) -> str:
    """Add realistic noise to a business name."""
    variations = [
        lambda s: s,
        lambda s: s.upper(),
        lambda s: s.lower(),
        lambda s: s.replace("&", "and"),
        lambda s: s.replace("Inc", "Incorporated"),
        lambda s: s.replace("Co", "Company"),
        lambda s: s + " LLC",
        lambda s: s.replace(" ", "  "),  # extra spaces
        lambda s: " " + s + " ",  # leading/trailing spaces
        lambda s: s.replace(",", ""),  # remove commas
    ]
    return random.choice(variations)(name)


def _add_address_noise(addr: str) -> str:
    """Add realistic noise to an address."""
    variations = [
        lambda s: s,
        lambda s: s.upper(),
        lambda s: s.lower(),
        lambda s: s.replace("Street", "St"),
        lambda s: s.replace("Avenue", "Ave"),
        lambda s: s.replace("Road", "Rd"),
        lambda s: s.replace("Boulevard", "Blvd"),
        lambda s: s.replace("Drive", "Dr"),
        lambda s: s.replace(",", ""),
        lambda s: s.replace("Suite ", "Ste "),
        lambda s: s.replace("Rue", "R."),
    ]
    return random.choice(variations)(addr)


def generate_synthetic_er_data(
    n_entities: int = 50,
    match_rate: float = 0.6,
    seed: int = 42,
):
    """Generate synthetic entity resolution data."""
    random.seed(seed)

    entities = []
    for i in range(n_entities):
        country = random.choice(COUNTRIES)
        name = random.choice(BUSINESS_NAMES)
        if country == "US":
            addr = random.choice(ADDRESSES_US)
        elif country == "India":
            addr = random.choice(ADDRESSES_INDIA)
        else:
            addr = random.choice(ADDRESSES_FRANCE)
        entities.append({
            "name": name,
            "address": addr,
            "country": country,
        })

    # Generate sources
    s1_rows, s2_rows, s3_rows = [], [], []
    gt_rows = []

    for i, ent in enumerate(entities):
        s1_id = f"s1_{i:04d}"
        s1_rows.append({
            "entity_id": s1_id,
            "business_name": ent["name"],
            "business_address": ent["address"],
            "country": ent["country"],
        })

        # Decide if this entity has matches
        if random.random() < match_rate:
            # Create S2 match
            s2_id = f"s2_{i:04d}"
            s2_rows.append({
                "entity_id": s2_id,
                "business_name": _add_noise(ent["name"]),
                "business_address": _add_address_noise(ent["address"]),
                "country": ent["country"],
            })
            gt_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": s2_id,
            })

            # Sometimes also S3 match
            if random.random() < 0.3:
                s3_id = f"s3_{i:04d}"
                s3_rows.append({
                    "entity_id": s3_id,
                    "business_name": _add_noise(ent["name"]),
                    "business_address": _add_address_noise(ent["address"]),
                    "country": ent["country"],
                })
                gt_rows[-1]["matched_entity_ids"] += f",{s3_id}"
        else:
            # Singleton — no match in GT
            gt_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": "",
            })

    # Add some non-matching entities to S2 and S3
    for i in range(n_entities, n_entities + 20):
        country = random.choice(COUNTRIES)
        name = random.choice(BUSINESS_NAMES)
        if country == "US":
            addr = random.choice(ADDRESSES_US)
        elif country == "India":
            addr = random.choice(ADDRESSES_INDIA)
        else:
            addr = random.choice(ADDRESSES_FRANCE)

        s2_rows.append({
            "entity_id": f"s2_{i:04d}",
            "business_name": name,
            "business_address": addr,
            "country": country,
        })
        s3_rows.append({
            "entity_id": f"s3_{i:04d}",
            "business_name": name,
            "business_address": addr,
            "country": country,
        })

    import pandas as pd
    return {
        "source1": pd.DataFrame(s1_rows),
        "source2": pd.DataFrame(s2_rows),
        "source3": pd.DataFrame(s3_rows),
        "ground_truth": pd.DataFrame(gt_rows),
    }


def save_synthetic_data(output_dir: str = "dataset", n_entities: int = 50, seed: int = 42):
    """Generate and save synthetic data in the expected directory structure."""
    data = generate_synthetic_er_data(n_entities=n_entities, seed=seed)

    # Train data (first 70%)
    split_idx = int(n_entities * 0.7)

    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Use first 70% of S1 as train, rest as test
    s1_all = data["source1"]
    s1_train = s1_all.iloc[:split_idx].reset_index(drop=True)
    s1_test = s1_all.iloc[split_idx:].reset_index(drop=True)

    train_s1_ids = set(s1_train["entity_id"])
    test_s1_ids = set(s1_test["entity_id"])

    # Split S2/S3 similarly (matching entities go with their S1)
    gt = data["ground_truth"]
    gt_train = gt[gt["source1_entity_id"].isin(train_s1_ids)].reset_index(drop=True)
    # Test has no ground truth in real scenario

    # Determine which S2/S3 IDs are matched to train vs test
    train_matched_ids = set()
    for _, row in gt_train.iterrows():
        for mid in str(row["matched_entity_ids"]).split(","):
            mid = mid.strip()
            if mid:
                train_matched_ids.add(mid)

    test_matched_ids = set()
    gt_test = gt[gt["source1_entity_id"].isin(test_s1_ids)]
    for _, row in gt_test.iterrows():
        for mid in str(row["matched_entity_ids"]).split(","):
            mid = mid.strip()
            if mid:
                test_matched_ids.add(mid)

    s2 = data["source2"]
    s3 = data["source3"]

    # S2/S3: include all entities (both matched and unmatched)
    # For train, include all S2/S3 (they need to exist for blocking)
    # For test, also include all S2/S3

    s1_train.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    s2.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    s3.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    gt_train.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    s1_test.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    print(f"Generated synthetic data in {output_dir}/")
    print(f"  Train: {len(s1_train)} S1, {len(s2)} S2, {len(s3)} S3, {len(gt_train)} GT rows")
    print(f"  Test:  {len(s1_test)} S1, {len(s2)} S2, {len(s3)} S3")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--n-entities", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    save_synthetic_data(args.output_dir, args.n_entities, args.seed)
