"""
Generates a per-circuit lap count table from results.csv, races.csv, and circuits.csv.

Usage:
    python server/helper/circuit_laps.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]  # points to /server
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_FILE = BASE_DIR / "helper" / "circuit_laps.json"

# --- Load CSVs ---
results = pd.read_csv(DATA_DIR / "results.csv")
races = pd.read_csv(DATA_DIR / "races.csv")
circuits = pd.read_csv(DATA_DIR / "circuits.csv")

# --- Step 1: get per-race lap counts ---
race_laps = results.groupby("raceId")["laps"].max().reset_index(name="total_laps")

# --- Step 2: merge with races and circuits ---
merged = (
    race_laps
    .merge(races[["raceId", "circuitId", "year", "name"]], on="raceId", how="left")
    .merge(circuits[["circuitId", "name", "country"]], on="circuitId", how="left", suffixes=("_race", "_circuit"))
)

# --- Step 3: filter outliers (drop early-terminated races) ---
def remove_outliers(x):
    mean = x["total_laps"].mean()
    mask = x["total_laps"].between(mean * 0.85, mean * 1.15)
    return x[mask]

filtered = merged.groupby("circuitId", group_keys=False).apply(remove_outliers)

# --- Step 4: compute average laps per circuit ---
circuit_laps = (
    filtered.groupby(["circuitId", "name_circuit", "country"])["total_laps"]
    .mean()
    .round()
    .reset_index()
    .rename(columns={"total_laps": "avgLaps"})
    .sort_values("avgLaps", ascending=False)
)

# --- Step 5: save as JSON for main.py to import later ---
circuit_laps.to_json(OUTPUT_FILE, orient="records", indent=2)
print(f"✅ Saved average laps per circuit to {OUTPUT_FILE}")
print(circuit_laps.head())
