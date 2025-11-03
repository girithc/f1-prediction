# server/main.py
import os
import json
import math
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from joblib import load

# =====================================================
# Config
# =====================================================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "artifacts"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://localhost,https://*.web.app,https://*.firebaseapp.com",
)

# Artifacts
REGRESSOR_PATH = MODEL_DIR / "finish_regressor_xgb.pkl"   # XGB pipeline (with preprocessing)
TOP3_CLS_PATH = MODEL_DIR / "top3_classifier_lr.pkl"      # LogisticRegression pipeline (with preprocessing)

# Helper metadata files produced by your scripts
CIRCUIT_LAPS_PATH = HERE / "helper" / "circuit_laps.json"        # [{circuitId, name_circuit, country, avgLaps}]
OVERTAKE_INDEX_PATH = HERE / "helper" / "overtake_index.json"    # [{circuitId, overtakeIndex}] (higher = easier)

# If your helper produces "overtakeIndex" as 'ease', flip to 'difficulty' expected by the model
# circuit_overtake_difficulty = 1 - normalized(overtakeIndex)
OVERTAKE_HIGHER_IS_EASIER = True  # set False if your JSON already stores "difficulty" (higher = harder)


# =====================================================
# Schemas (I/O)
# =====================================================
class PitStop(BaseModel):
    lap: int = Field(..., ge=1, description="Lap number of the stop")
    durationMs: int = Field(..., ge=1000, le=100000, description="Pit stop duration in milliseconds")

class PredictRequest(BaseModel):
    circuitId: int | str
    gridPosition: int = Field(..., ge=1, le=20)
    pitPlan: List[PitStop]
    driverId: Optional[str] = None  # not used by the model, included for completeness

    @validator("pitPlan")
    def sort_pits(cls, v: List[PitStop]) -> List[PitStop]:
        return sorted(v, key=lambda p: p.lap)

class FeatureImpact(BaseModel):
    name: str
    impact: float
    direction: Optional[str] = None

class PredictResponse(BaseModel):
    prediction: Dict[str, float]
    top3: Dict[str, float]
    perPitEffects: Optional[List[Dict[str, float]]] = None
    explanation: Optional[Dict[str, List[FeatureImpact]]] = None
    modelVersion: str

class CompareScenario(BaseModel):
    id: str
    circuitId: int | str
    gridPosition: int = Field(..., ge=1, le=20)
    pitPlan: List[PitStop]

class CompareRequest(BaseModel):
    scenarios: List[CompareScenario]

class CompareResult(BaseModel):
    scenarioId: str
    finishP50: float
    intervalWidth: float
    top3Probability: float
    robustnessScore: float

class CompareResponse(BaseModel):
    results: List[CompareResult]
    recommendedScenarioId: str
    modelVersion: str


# =====================================================
# App init + CORS
# =====================================================
app = FastAPI(title="F1 Strategy Prediction API", version=MODEL_VERSION)

origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Load artifacts + metadata at startup
# =====================================================
@dataclass
class Artifacts:
    regressor: Any = None
    top3_clf: Any = None
    circuit_laps: list = None  # [{circuitId, name_circuit, country, avgLaps}]
    overtake_difficulty: dict = None  # {circuitId -> circuit_overtake_difficulty}

ART = Artifacts()

def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def _build_overtake_difficulty_map(overtake_json: list | None) -> Dict[int, float]:
    """
    Input JSON (from helper) format:
      [{ "circuitId": <int>, "overtakeIndex": <float> }, ...]
      where overtakeIndex is normalized 0..1 (higher = easier) by default.

    We convert to difficulty expected by training:
      circuit_overtake_difficulty = 1 - overtakeIndex   (higher = harder)
    """
    if not overtake_json:
        return {}

    # Detect if circuitId is numeric; coerce to int keys
    out: Dict[int, float] = {}
    vals = [row.get("overtakeIndex") for row in overtake_json if "overtakeIndex" in row]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    rng = (vmax - vmin) or 1.0

    for row in overtake_json:
        cid_raw = row.get("circuitId")
        try:
            cid = int(cid_raw)
        except Exception:
            # If it's not an int (unlikely), use hashable fallback mapping
            cid = int(str(cid_raw).split("_")[0]) if str(cid_raw).split("_")[0].isdigit() else hash(cid_raw)

        val = float(row.get("overtakeIndex", 0.5))
        # If helper forgot to normalize 0..1, normalize here defensively
        norm = (val - vmin) / rng
        if OVERTAKE_HIGHER_IS_EASIER:
            diff = 1.0 - norm
        else:
            diff = norm
        out[cid] = float(diff)

    return out

def _build_circuit_maps(circuit_laps_json: list | None):
    """
    Build two lookups:
      - circuit_meta: {circuitId -> {"name":..., "country":..., "avgLaps":...}}
      - name_to_id  : {normalized_name -> circuitId}
    """
    circuit_meta: Dict[int, Dict[str, Any]] = {}
    name_to_id: Dict[str, int] = {}
    if not circuit_laps_json:
        return circuit_meta, name_to_id

    for row in circuit_laps_json:
        cid_raw = row.get("circuitId")
        try:
            cid = int(cid_raw)
        except Exception:
            cid = int(str(cid_raw).split("_")[0]) if str(cid_raw).split("_")[0].isdigit() else hash(cid_raw)

        name = row.get("name_circuit") or row.get("name") or f"circuit_{cid}"
        country = row.get("country", "Unknown")
        avg_laps = int(row.get("avgLaps")) if row.get("avgLaps") is not None else None

        circuit_meta[cid] = {"name": name, "country": country, "avgLaps": avg_laps}
        name_to_id[name.strip().lower()] = cid

    return circuit_meta, name_to_id


@app.on_event("startup")
def _startup():
    # Artifacts
    if not REGRESSOR_PATH.exists():
        raise RuntimeError(f"Regressor artifact not found: {REGRESSOR_PATH}")
    if not TOP3_CLS_PATH.exists():
        raise RuntimeError(f"Top-3 classifier artifact not found: {TOP3_CLS_PATH}")

    ART.regressor = load(REGRESSOR_PATH)
    ART.top3_clf = load(TOP3_CLS_PATH)

    # Metadata JSONs
    circuit_laps_json = _load_json(CIRCUIT_LAPS_PATH) or []
    overtake_json = _load_json(OVERTAKE_INDEX_PATH) or []

    ART.circuit_laps = circuit_laps_json
    ART.overtake_difficulty = _build_overtake_difficulty_map(overtake_json)

    # Build circuit maps for quick access
    global CIRCUIT_META, NAME_TO_ID
    CIRCUIT_META, NAME_TO_ID = _build_circuit_maps(circuit_laps_json)


# =====================================================
# Feature builder (mirrors training-time features)
# =====================================================
NUMERIC_DEFAULTS = {
    "pit_count": 0,
    "pit_total_duration": 0,
    "pit_avg_duration": 0,
    "first_pit_lap": 0,
    "last_pit_lap": 0,
    "round": 1,  # not exposed; constant at inference (no leakage)
}

def _scenario_to_features(circuit_id: int | str, grid: int, pit_plan: List[PitStop]) -> pd.DataFrame:
    # Coerce circuit ID
    try:
        cid = int(circuit_id)
    except Exception:
        # if frontend passes a name string, try to map
        key = str(circuit_id).strip().lower()
        if key in NAME_TO_ID:
            cid = NAME_TO_ID[key]
        else:
            # fallback: attempt to parse leading integer, else hash (unlikely)
            cid = int(key.split("_")[0]) if key.split("_")[0].isdigit() else hash(key)

    # Circuit meta
    cmeta = CIRCUIT_META.get(cid, {"name": f"circuit_{cid}", "country": "Unknown", "avgLaps": None})
    country = cmeta["country"]

    # Pit aggregates
    pit_count = len(pit_plan)
    durations = [p.durationMs for p in pit_plan] if pit_plan else []
    laps = [p.lap for p in pit_plan] if pit_plan else []

    total_ms = int(np.sum(durations)) if durations else 0
    avg_ms = int(np.mean(durations)) if durations else 0
    first_lap = int(min(laps)) if laps else 0
    last_lap = int(max(laps)) if laps else 0

    # Overtake difficulty lookup; fallback to mean if missing
    od_map = ART.overtake_difficulty or {}
    od_values = list(od_map.values()) or [0.5]
    od_default = float(np.mean(od_values))
    circuit_overtake_difficulty = float(od_map.get(cid, od_default))

    row = pd.DataFrame([{
        # numeric
        "grid": grid,
        "pit_count": pit_count,
        "pit_total_duration": total_ms,
        "pit_avg_duration": avg_ms,
        "first_pit_lap": first_lap,
        "last_pit_lap": last_lap,
        "circuit_overtake_difficulty": circuit_overtake_difficulty,
        "round": NUMERIC_DEFAULTS["round"],
        # categorical
        "circuitId": cid,
        "country": country,
    }])

    return row


# =====================================================
# Inference helpers
# =====================================================
def _predict_finish_and_top3(circuit_id, grid, pit_plan: List[PitStop]):
    X_row = _scenario_to_features(circuit_id, grid, pit_plan)

    # Regressor (finish position)
    try:
        finish_pred = float(ART.regressor.predict(X_row)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regression prediction failed: {e}")

    # Top-3 probability
    try:
        prob = ART.top3_clf.predict_proba(X_row)[0, 1]
        top3_prob = float(prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top-3 probability failed: {e}")

    # Simple uncertainty proxy (use quantiles if you trained quantile models; here a heuristic)
    # Clamp to plausible positions [1, 20]
    p50 = max(1.0, min(20.0, finish_pred))
    # heuristic interval width based on historical residual scale; feel free to replace with conformal intervals
    interval_width = 4.0
    p10 = max(1.0, p50 - interval_width/2)
    p90 = min(20.0, p50 + interval_width/2)

    return {
        "prediction": {"finishP50": p50, "finishP10": float(p10), "finishP90": float(p90)},
        "top3": {"probability": top3_prob},
    }


def _robustness_score(p50: float, interval_width: float, top3_prob: float) -> float:
    # Higher when intervals are narrower and top3 is higher; lightly reward better (lower) p50
    # Normalize rough scales
    iw_score = max(0.0, 1.0 - (interval_width / 10.0))        # narrower is better
    top3_score = float(top3_prob)                              # already 0..1
    rank_score = max(0.0, 1.0 - (p50 - 1.0) / 19.0)            # 1.0 at P1, ~0 at P20
    return float(0.4 * iw_score + 0.4 * top3_score + 0.2 * rank_score)


# =====================================================
# Routes
# =====================================================
@app.get("/healthz")
def healthz():
    ok = ART.regressor is not None and ART.top3_clf is not None
    return {"status": "ok" if ok else "error", "modelVersion": MODEL_VERSION}

@app.get("/metadata")
def metadata():
    # Expose circuits with avgLaps plus the overtake difficulty for convenience
    circuits = []
    od_map = ART.overtake_difficulty or {}
    for row in (ART.circuit_laps or []):
        cid_raw = row.get("circuitId")
        try:
            cid = int(cid_raw)
        except Exception:
            cid = int(str(cid_raw).split("_")[0]) if str(cid_raw).split("_")[0].isdigit() else hash(cid_raw)
        circuits.append({
            "circuitId": cid,
            "name": row.get("name_circuit") or row.get("name") or f"circuit_{cid}",
            "country": row.get("country", "Unknown"),
            "avgLaps": row.get("avgLaps"),
            "overtakeDifficulty": float(od_map.get(cid, 0.5))
        })

    return {
        "circuits": circuits,
        "modelVersion": MODEL_VERSION
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    out = _predict_finish_and_top3(req.circuitId, req.gridPosition, req.pitPlan)
    return {
        "prediction": out["prediction"],
        "top3": out["top3"],
        "perPitEffects": None,   # populate if you add per-pit modeling
        "explanation": None,     # populate with SHAP if desired
        "modelVersion": MODEL_VERSION,
    }

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    results: List[CompareResult] = []
    for sc in req.scenarios:
        out = _predict_finish_and_top3(sc.circuitId, sc.gridPosition, sc.pitPlan)
        p50 = out["prediction"]["finishP50"]
        p10 = out["prediction"]["finishP10"]
        p90 = out["prediction"]["finishP90"]
        interval_width = float(p90 - p10)
        top3_prob = out["top3"]["probability"]
        robust = _robustness_score(p50, interval_width, top3_prob)
        results.append(CompareResult(
            scenarioId=sc.id,
            finishP50=float(p50),
            intervalWidth=float(interval_width),
            top3Probability=float(top3_prob),
            robustnessScore=float(robust),
        ))

    # Recommend the scenario with max robustness score
    best = max(results, key=lambda r: r.robustnessScore) if results else None
    return CompareResponse(
        results=results,
        recommendedScenarioId=best.scenarioId if best else "",
        modelVersion=MODEL_VERSION,
    )
