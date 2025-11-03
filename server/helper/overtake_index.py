import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_FILE = BASE_DIR / "helper" / "overtake_index.json"

results = pd.read_csv(DATA_DIR / "results.csv")
races = pd.read_csv(DATA_DIR / "races.csv")

# Merge race and circuit info
merged = results.merge(races[["raceId", "circuitId", "year"]], on="raceId", how="left")

# Exclude DNFs (statusId != 1 in Ergast usually means not finished)
merged = merged[merged["statusId"] == 1]

# Compute positions gained/lost
merged["pos_gain"] = merged["grid"] - merged["positionOrder"]

# Aggregate by circuit
overtake = (
    merged.groupby("circuitId")["pos_gain"]
    .mean()
    .reset_index()
    .rename(columns={"pos_gain": "overtakeIndex"})
)

# Normalize 0–1 (optional)
overtake["overtakeIndex"] = (overtake["overtakeIndex"] - overtake["overtakeIndex"].min()) / (
    overtake["overtakeIndex"].max() - overtake["overtakeIndex"].min()
)

overtake.to_json(OUTPUT_FILE, orient="records", indent=2)
print(f"✅ Saved overtake index to {OUTPUT_FILE}")
print(overtake.head())
