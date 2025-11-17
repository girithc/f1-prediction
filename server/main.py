import os
import json
import math
import sys  # <--- Added sys for the fix
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from joblib import load
import uvicorn
import xgboost as xgb  # Required for DMatrix inside BoosterWrapper

# =====================================================
# Config
# =====================================================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "artifacts"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2.2.1")

API_BASE_URL = os.getenv("API_BASE_URL", "http://0.0.0.0")

REGRESSOR_PATH = MODEL_DIR / "finish_regressor_xgb.pkl"

CIRCUIT_LAPS_PATH = HERE / "helper" / "circuit_laps.json"
OVERTAKE_INDEX_PATH = HERE / "helper" / "overtake_index.json"

OVERTAKE_HIGHER_IS_EASIER = True

DEBUG_PRED = os.getenv("DEBUG_PRED", "0") == "1"
SQUASH_SCALE = float(os.getenv("SQUASH_SCALE", "3.0"))
SQUASH_BIAS  = float(os.getenv("SQUASH_BIAS",  "0.0"))
INTERVAL_WIDTH = float(os.getenv("INTERVAL_WIDTH", "4.0"))
ROUND_DEFAULT  = int(os.getenv("ROUND_DEFAULT", "1"))

# =====================================================
# ⚠️ CRITICAL: Class Definition for Pickle ⚠️
# =====================================================
class BoosterWrapper:
    def __init__(self, booster, preprocessor):
        self.booster = booster
        self.preprocessor = preprocessor
    def predict(self, X):
        Xt = self.preprocessor.transform(X)
        # We must wrap transformed data in DMatrix for the raw booster
        return self.booster.predict(xgb.DMatrix(Xt))

# =====================================================
# Schemas (I/O)
# =====================================================
class PitStop(BaseModel):
    lap: int = Field(..., ge=1)
    durationMs: int = Field(..., ge=1000, le=100000)

class PredictRequest(BaseModel):
    circuitId: Union[int, str]
    gridPosition: int = Field(..., ge=1, le=20)
    pitPlan: List[PitStop]

    carPerformanceIndex: Optional[float] = Field(None, ge=0.0, le=1.0)
    avgTireScore: Optional[float] = Field(None, ge=0.0, le=3.0)
    round: Optional[int] = Field(None, ge=1, le=25)

    driverId: Optional[str] = None

    @validator("pitPlan")
    def sort_pits(cls, v: List[PitStop]) -> List[PitStop]:
        return sorted(v, key=lambda p: p.lap)

class FeatureImpact(BaseModel):
    name: str
    impact: float
    direction: Optional[str] = None

class PredictResponse(BaseModel):
    prediction: Dict[str, float]
    top3: Dict[str, Any]
    positionProbs: Optional[Dict[str, float]] = None
    perPitEffects: Optional[List[Dict[str, float]]] = None
    explanation: Optional[Dict[str, List[FeatureImpact]]] = None
    modelVersion: str

class CompareScenario(BaseModel):
    id: str
    circuitId: Union[int, str]
    gridPosition: int = Field(..., ge=1, le=20)
    pitPlan: List[PitStop]
    carPerformanceIndex: Optional[float] = Field(None, ge=0.0, le=1.0)
    avgTireScore: Optional[float] = Field(None, ge=0.0, le=3.0)
    round: Optional[int] = Field(None, ge=1, le=25)

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
# App init
# =====================================================
app = FastAPI(title="F1 Strategy Prediction API", version=MODEL_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Artifacts + helpers
# =====================================================
@dataclass
class Artifacts:
    regressor: Any = None
    circuit_laps: list = None
    overtake_difficulty: dict = None
    grid_feature_name: str = "grid"

ART = Artifacts()
CIRCUIT_META: Dict[int, Dict[str, Any]] = {}
NAME_TO_ID: Dict[str, int] = {}

def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def _require_int_circuit_id(val: Any) -> int:
    if val is None:
        raise ValueError("circuitId is None")
    if isinstance(val, int):
        return val
    s = str(val).strip()
    if s.isdigit():
        return int(s)
    raise ValueError(f"Invalid circuitId: {val}")

def _build_overtake_difficulty_map(overtake_json: Optional[list]) -> Dict[int, float]:
    if not overtake_json:
        return {}
    vals = [row.get("overtakeIndex") for row in overtake_json if "overtakeIndex" in row]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    rng = (vmax - vmin) or 1.0
    out: Dict[int, float] = {}
    for row in overtake_json:
        try:
            cid = _require_int_circuit_id(row.get("circuitId"))
        except ValueError:
            continue
        val = float(row.get("overtakeIndex", 0.5))
        norm = (val - vmin) / rng
        diff = 1.0 - norm if OVERTAKE_HIGHER_IS_EASIER else norm
        out[cid] = float(diff)
    return out

def _build_circuit_maps(circuit_laps_json: Optional[list]):
    circuit_meta: Dict[int, Dict[str, Any]] = {}
    name_to_id: Dict[str, int] = {}
    if not circuit_laps_json:
        return circuit_meta, name_to_id
    for row in circuit_laps_json:
        try:
            cid = _require_int_circuit_id(row.get("circuitId"))
        except ValueError:
            continue
        name = row.get("name_circuit") or row.get("name") or f"circuit_{cid}"
        country = row.get("country", "Unknown")
        avg_laps = row.get("avgLaps")
        try:
            avg_laps = float(avg_laps) if avg_laps is not None else None
        except Exception:
            avg_laps = None
        circuit_meta[cid] = {"name": name, "country": country, "avgLaps": avg_laps}
        name_to_id[name.strip().lower()] = cid
    return circuit_meta, name_to_id

def _resolve_grid_feature_name() -> str:
    try:
        names = ART.regressor.preprocessor.feature_names_in_
        return "grid" if names is None else ("grid" if "grid" in names else names[0])
    except AttributeError:
        pass

    names = getattr(ART.regressor, "feature_names_in_", None)
    if names is None:
        return "grid"
    candidates = [
        "grid", "gridPosition", "start_grid", "start_pos", "startPosition",
        "qualy_pos", "qualyPosition", "startGrid"
    ]
    for c in candidates:
        if c in names:
            return c
    return "grid"

def _align_to_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        names = ART.regressor.preprocessor.feature_names_in_
    except AttributeError:
        names = getattr(ART.regressor, "feature_names_in_", None)
    
    if names is None:
        return df
    return df.reindex(columns=list(names), fill_value=0)

# =====================================================
# Startup
# =====================================================
@app.on_event("startup")
def _startup():
    if not REGRESSOR_PATH.exists():
        raise RuntimeError(f"Regressor artifact not found: {REGRESSOR_PATH}")
    
    # =========================================================
    # ⚠️ PATCH: Fix Namespace Error for Pickle + Uvicorn ⚠️
    # =========================================================
    # When using uvicorn --reload, the main module is often named
    # '__mp_main__', but the pickle file expects 'BoosterWrapper'
    # to be in '__main__'. We must inject it into both.
    # =========================================================
    import __main__
    setattr(__main__, "BoosterWrapper", BoosterWrapper)

    if "__mp_main__" in sys.modules:
        setattr(sys.modules["__mp_main__"], "BoosterWrapper", BoosterWrapper)
    # =========================================================

    ART.regressor = load(REGRESSOR_PATH)

    circuit_laps_json = _load_json(CIRCUIT_LAPS_PATH) or []
    overtake_json     = _load_json(OVERTAKE_INDEX_PATH) or []

    ART.circuit_laps = circuit_laps_json
    ART.overtake_difficulty = _build_overtake_difficulty_map(overtake_json)

    global CIRCUIT_META, NAME_TO_ID
    CIRCUIT_META, NAME_TO_ID = _build_circuit_maps(circuit_laps_json)

    ART.grid_feature_name = _resolve_grid_feature_name()
    
    try:
        expected = ART.regressor.preprocessor.feature_names_in_
        print(f"Model expects features: {list(expected)}")
    except AttributeError:
        print("Could not retrieve feature names from preprocessor.")

# =====================================================
# Request → Features
# =====================================================
NUMERIC_DEFAULTS = {
    "pit_count": 0,
    "pit_total_duration": 0,
    "pit_avg_duration": 0,
    "first_pit_lap": 0,
    "last_pit_lap": 0,
    "round": ROUND_DEFAULT,
    "carPerformanceIndex": 0.5,
    "avgTireScore": 1.8,
}

def _resolve_circuit_id(circuit_id: Union[int, str]) -> int:
    try:
        return int(circuit_id)
    except Exception:
        key = str(circuit_id).strip().lower()
        if key in NAME_TO_ID:
            return NAME_TO_ID[key]
        if key.isdigit():
            return int(key)
        raise HTTPException(status_code=422, detail=f"Invalid circuitId: {circuit_id}")

def _scenario_to_features(
    circuit_id: Union[int, str],
    grid: int,
    pit_plan: List[PitStop],
    car_perf: Optional[float],
    avg_tire: Optional[float],
    round_override: Optional[int],
) -> pd.DataFrame:
    cid = _resolve_circuit_id(circuit_id)
    cmeta = CIRCUIT_META.get(cid, {"name": f"circuit_{cid}", "country": "Unknown", "avgLaps": None})
    country = cmeta["country"]

    pit_count = len(pit_plan)
    durations = [p.durationMs for p in pit_plan] if pit_plan else []
    laps = [p.lap for p in pit_plan] if pit_plan else []

    total_ms = int(np.sum(durations)) if durations else 0
    avg_ms = int(np.mean(durations)) if durations else 0
    first_lap = int(min(laps)) if laps else 0
    last_lap  = int(max(laps)) if laps else 0

    od_map = ART.overtake_difficulty or {}
    od_values = list(od_map.values()) or [0.5]
    od_default = float(np.mean(od_values))
    circuit_overtake_difficulty = float(od_map.get(cid, od_default))

    round_val = int(round_override) if round_override is not None else NUMERIC_DEFAULTS["round"]
    car_pi    = float(car_perf) if car_perf is not None else NUMERIC_DEFAULTS["carPerformanceIndex"]
    avg_tire_score = float(avg_tire) if avg_tire is not None else NUMERIC_DEFAULTS["avgTireScore"]

    grid_raw = max(1, min(20, int(grid)))

    tire_stints = pit_count + 1
    avg_pit_ms_val = avg_ms 

    # Calculate season_progress (assuming roughly 22 races if not provided)
    # This ensures the feature exists and isn't always 0
    season_prog = (round_val - 1) / 21.0 

    base = {
        ART.grid_feature_name: grid_raw, 
        "pit_count": pit_count,
        "pit_total_duration": total_ms,
        "pit_avg_duration": avg_ms,
        "first_pit_lap": first_lap,
        "last_pit_lap": last_lap,
        "circuit_overtake_difficulty": circuit_overtake_difficulty,
        "round": round_val,
        "circuitId": cid,
        "country": country,
        "carPerformanceIndex": car_pi,
        "avgTireScore": avg_tire_score,
        "tireStints": tire_stints, 
        "avgPitMs": avg_pit_ms_val,
        "first_stop_delta": float(first_lap) / float(cmeta.get("avgLaps") or 60.0) if first_lap > 0 else 0.0,
        "tire_aggr_index": (tire_stints / total_ms) if total_ms > 0 else 0.0,
        "season_progress": season_prog # <--- Added this
    }
    row = pd.DataFrame([base])

    row = _align_to_model_columns(row)
    return row# =====================================================
# Post-processing
# =====================================================
def _squash_to_1_20(x: float, scale: float = SQUASH_SCALE, bias: float = SQUASH_BIAS) -> float:
    # Sigmoid squash to ensure we stay in 1..20 range
    z = (x + bias) / max(1e-6, scale)
    return 1.0 + 19.0 * (1.0 / (1.0 + math.exp(-z)))

_Z_90 = 1.2815515655446004

def _infer_sigma_from_interval(p10: float, p90: float) -> float:
    band = max(1e-6, p90 - p10)
    return max(1e-3, band / (2.0 * _Z_90))

def _phi_cdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / (max(1e-9, sigma) * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def _position_distribution(p50: float, p10: float, p90: float) -> Dict[str, float]:
    mu = float(p50)
    sigma = _infer_sigma_from_interval(float(p10), float(p90))
    probs = []
    for k in range(1, 21):
        lo = -float("inf") if k == 1 else (k - 0.5)
        hi =  float("inf") if k == 20 else (k + 0.5)
        p = _phi_cdf(hi, mu, sigma) - _phi_cdf(lo, mu, sigma)
        probs.append(max(0.0, p))
    s = sum(probs) or 1.0
    probs = [p / s for p in probs]
    return {f"P{k}": probs[k-1] for k in range(1, 21)}


def _predict_distribution(
    circuit_id,
    grid,
    pit_plan: List[PitStop],
    car_perf: Optional[float],
    avg_tire: Optional[float],
    round_override: Optional[int],
):
    X_row = _scenario_to_features(circuit_id, grid, pit_plan, car_perf, avg_tire, round_override)

    if DEBUG_PRED:
        print("\n--- DEBUG: Inference row ---")
        print(list(X_row.columns))
        print(X_row.to_dict(orient="records"))

    try:
        finish_pred_raw = float(ART.regressor.predict(X_row)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regression prediction failed: {e}")

    # --- FIX: REMOVE SQUASH, USE CLIP ---
    # The model predicts 1..20 directly. We just ensure it stays in bounds.
    p50 = max(1.0, min(20.0, finish_pred_raw))
    # ------------------------------------
    
    p10 = max(1.0, p50 - INTERVAL_WIDTH / 2)
    p90 = min(20.0, p50 + INTERVAL_WIDTH / 2)

    pos_probs = _position_distribution(p50, p10, p90)
    top3_prob = float(pos_probs["P1"] + pos_probs["P2"] + pos_probs["P3"])

    return {
        "prediction": {
            "finishP50": float(p50),
            "finishP10": float(p10),
            "finishP90": float(p90)
        },
        "top3": {"probability": top3_prob, "source": "distribution"},
        "positionProbs": pos_probs
    }



def _robustness_score(p50: float, interval_width: float, top3_prob: float) -> float:
    iw_score = max(0.0, 1.0 - (interval_width / 10.0))
    top3_score = float(top3_prob)
    rank_score = max(0.0, 1.0 - (p50 - 1.0) / 19.0)
    return float(0.4 * iw_score + 0.4 * top3_score + 0.2 * rank_score)

# =====================================================
# Endpoints
# =====================================================
@app.get("/healthz")
def healthz():
    ok = ART.regressor is not None
    return {"status": "ok" if ok else "error", "modelVersion": MODEL_VERSION}

@app.get("/metadata")
def metadata():
    circuits = []
    od_map = ART.overtake_difficulty or {}
    for row in (ART.circuit_laps or []):
        try:
            cid = _require_int_circuit_id(row.get("circuitId"))
        except ValueError:
            continue
        circuits.append({
            "circuitId": cid,
            "name": row.get("name_circuit") or row.get("name") or f"circuit_{cid}",
            "country": row.get("country", "Unknown"),
            "avgLaps": row.get("avgLaps"),
            "overtakeDifficulty": float(od_map.get(cid, 0.5))
        })
    return {"circuits": circuits, "modelVersion": MODEL_VERSION}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    out = _predict_distribution(
        circuit_id=req.circuitId,
        grid=req.gridPosition,
        pit_plan=req.pitPlan,
        car_perf=req.carPerformanceIndex,
        avg_tire=req.avgTireScore,
        round_override=req.round,
    )
    return {
        "prediction": out["prediction"],
        "top3": out["top3"],
        "positionProbs": out["positionProbs"],
        "perPitEffects": None,
        "explanation": None,
        "modelVersion": MODEL_VERSION
    }

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    results: List[CompareResult] = []
    for sc in req.scenarios:
        out = _predict_distribution(
            circuit_id=sc.circuitId,
            grid=sc.gridPosition,
            pit_plan=sc.pitPlan,
            car_perf=sc.carPerformanceIndex,
            avg_tire=sc.avgTireScore,
            round_override=sc.round,
        )
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
    best = max(results, key=lambda r: r.robustnessScore) if results else None
    return CompareResponse(
        results=results,
        recommendedScenarioId=best.scenarioId if best else "",
        modelVersion=MODEL_VERSION
    )

@app.get("/introspect")
def introspect():
    names = None
    try:
        names = ART.regressor.preprocessor.feature_names_in_
    except AttributeError:
        names = getattr(ART.regressor, "feature_names_in_", None)

    return {
        "modelVersion": MODEL_VERSION,
        "gridFeatureName": ART.grid_feature_name,
        "feature_names_in_": list(map(str, names)) if names is not None else None
    }

class WhatIfRequest(PredictRequest):
    pass

@app.post("/whatif")
def whatif(req: WhatIfRequest):
    rows = []
    for g in range(1, 21):
        out = _predict_distribution(
            circuit_id=req.circuitId,
            grid=g,
            pit_plan=req.pitPlan,
            car_perf=req.carPerformanceIndex,
            avg_tire=req.avgTireScore,
            round_override=req.round,
        )
        rows.append({"grid": g, "finishP50": out["prediction"]["finishP50"]})
    return {
        "modelVersion": MODEL_VERSION,
        "gridFeatureName": ART.grid_feature_name,
        "series": rows
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting FastAPI server on {API_BASE_URL} (Port: {port})")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)