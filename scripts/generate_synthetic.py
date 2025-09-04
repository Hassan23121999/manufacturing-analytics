#!/usr/bin/env python3
"""
Synthetic manufacturing datasets generator
- parts.csv / .parquet
- production.csv / .parquet
- quality.csv / .parquet
- maintenance.csv / .parquet

Now includes controlled NULL injection to test ETL cleaning.
"""

import argparse, os, random, math
from datetime import timedelta
import numpy as np
import pandas as pd

# -------------------------
# Global configuration
# -------------------------
MATERIALS = [
    # name       cycle_coef  energy_coef  tol_baseline_um
    ("Aluminum", 0.85,       0.040,       60),
    ("Steel",    1.10,       0.055,       40),
    ("ABS",      0.70,       0.030,       80),
    ("Brass",    0.95,       0.048,       50),
    ("Titanium", 1.40,       0.070,       35),
]
FAILURE_MODES = ["Bearing", "Thermal", "Vibration", "Hydraulic", "Electrical"]

# -------------------------
# Helpers
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def introduce_nulls(df: pd.DataFrame, col_probs: dict[str, float]) -> pd.DataFrame:
    """
    Randomly set some values to NaN in specified columns.
    col_probs example: {"material": 0.02, "energy_kwh": 0.03}
    """
    n = len(df)
    if n == 0:
        return df
    for col, p in col_probs.items():
        if col in df.columns and 0.0 < p <= 1.0:
            mask = np.random.rand(n) < p
            df.loc[mask, col] = np.nan
    return df

# -------------------------
# Data builders
# -------------------------
def make_parts(n_parts: int, start_id: int = 1) -> pd.DataFrame:
    part_ids = [f"P{start_id + i:06d}" for i in range(n_parts)]
    mats_idx = np.random.choice(range(len(MATERIALS)), size=n_parts,
                                p=[0.28, 0.28, 0.22, 0.12, 0.10])
    # geometry: volume ~ lognormal; SA ~ volume^(2/3) * jitter
    volume = np.random.lognormal(mean=9.4, sigma=0.55, size=n_parts)  # ~12k median
    sa = (volume ** (2.0/3.0)) * np.random.uniform(0.9, 1.15, size=n_parts)
    complexity = np.clip(np.random.beta(2.0, 3.0, size=n_parts), 0.02, 0.98)

    # tolerance: depends on material baseline; slightly tighter for higher complexity on average
    tol = []
    for m in mats_idx:
        base = MATERIALS[m][3]
        t = np.random.normal(loc=base - 12*complexity.mean(), scale=12)
        tol.append(np.clip(t, 15, 120))
    tol = np.array(tol)

    rev = np.ones(n_parts, dtype=int)
    created_at = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        np.random.randint(0, 60, size=n_parts), unit="D"
    )

    parts = pd.DataFrame({
        "part_id": part_ids,
        "material": [MATERIALS[m][0] for m in mats_idx],
        "volume_mm3": np.round(volume, 0),
        "surface_area_mm2": np.round(sa, 0),
        "tolerance_um": np.round(tol, 0),
        "complexity_idx": np.round(complexity, 3),
        "rev": rev,
        "created_at": created_at
    })
    return parts

def make_production(parts: pd.DataFrame,
                    events_per_part: tuple[int, int],
                    n_machines: int,
                    start_ts: str) -> pd.DataFrame:
    start = pd.to_datetime(start_ts)
    rows = []
    machine_ids = [f"M{i:03d}" for i in range(1, n_machines+1)]
    operator_ids = [f"O{i:03d}" for i in range(1, max(10, n_machines//2)+1)]

    mat_to_idx = {name: i for i, (name, *_rest) in enumerate(MATERIALS)}
    for _, r in parts.iterrows():
        k = random.randint(*events_per_part)
        m_idx = mat_to_idx[r.material]
        cycle_coef = MATERIALS[m_idx][1]
        energy_coef = MATERIALS[m_idx][2]
        

        for _ in range(k):
            ts = start + timedelta(minutes=random.randint(0, 60*24*30))
            machine = random.choice(machine_ids)
            operator = random.choice(operator_ids)

            # Cycle time model: increases with geometry, complexity, tighter tolerance
            geom_factor = (float(r.volume_mm3)**(1/3)) * 0.15 + (float(r.surface_area_mm2)**0.25) * 0.6
            tol_penalty = max(0.7, 1.8 - float(r.tolerance_um)/80.0)
            comp_penalty = 1.0 + 0.9*float(r.complexity_idx)
            base_cycle = 18 + 0.015*geom_factor
            cycle_time = base_cycle * cycle_coef * comp_penalty * tol_penalty
            cycle_time = np.random.normal(loc=cycle_time, scale=cycle_time*0.08)  # ~8% noise
            cycle_time = max(5.0, cycle_time)

            # Scrap probability: complexity & tight tolerance raise risk
            p_scrap = 0.02 + 0.15*float(r.complexity_idx) + max(0, (60 - float(r.tolerance_um))/200.0)
            p_scrap = float(np.clip(p_scrap + np.random.normal(0, 0.01), 0.0, 0.6))
            scrap_flag = np.random.rand() < p_scrap
            scrap_reason = random.choice(["Burr", "Warp", "Surface", "Dimensional", ""]) if scrap_flag else ""

            # Energy ~ cycle time × material factor (+ jitter)
            energy = max(0.2, energy_coef * cycle_time * np.random.uniform(0.9, 1.1))

            rows.append({
                "event_ts": ts,
                "part_id": r.part_id,
                "machine_id": machine,
                "cycle_time_sec": round(float(cycle_time), 2),
                "scrap_flag": scrap_flag,
                "scrap_reason": scrap_reason,
                "energy_kwh": round(float(energy), 3),
                "operator_id": operator
            })

    return pd.DataFrame(rows)

def make_quality(parts: pd.DataFrame,
                 inspections_per_part: tuple[int, int],
                 start_ts: str) -> pd.DataFrame:
    start = pd.to_datetime(start_ts)
    rows = []

    mat_sigma = {"Aluminum": 18, "Steel": 22, "ABS": 30, "Brass": 20, "Titanium": 24}

    for _, r in parts.iterrows():
        k = random.randint(*inspections_per_part)
        for _ in range(k):
            ts = start + timedelta(minutes=random.randint(0, 60*24*35))
            sigma = mat_sigma[r.material] * (0.6 + 1.2 * float(r.complexity_idx))
            dev = abs(np.random.normal(loc=0, scale=sigma))
            status = "PASS" if dev <= float(r.tolerance_um) else "FAIL"
            defect = "Dimension" if status == "FAIL" else random.choice(["Surface", "None", "Geometry"])

            rows.append({
                "inspection_ts": ts,
                "part_id": r.part_id,
                "dim_dev_um": round(float(dev), 1),
                "defect_type": defect,
                "status": status
            })

    return pd.DataFrame(rows)

def make_maintenance(prod: pd.DataFrame,
                     n_machines: int,
                     start_ts: str) -> pd.DataFrame:
    # Aggregate per machine to derive stress signal
    if len(prod) == 0:
        return pd.DataFrame(columns=["event_ts","machine_id","downtime_min","failure_mode","repair_cost"])

    df = prod.groupby("machine_id").agg(
        avg_cycle=("cycle_time_sec", "mean"),
        scrap_rate=("scrap_flag", "mean"),
        events=("part_id", "count")
    ).reset_index()

    # Stress combines normalized avg_cycle and scrap_rate
    stress = (df["avg_cycle"]/df["avg_cycle"].mean())*0.6 + \
             (df["scrap_rate"]/max(0.01, df["scrap_rate"].mean()))*0.4
    stress = stress.fillna(0.8)

    start = pd.to_datetime(start_ts)
    rows = []
    for i, row in df.iterrows():
        m = row["machine_id"]
        s = float(stress.iloc[i])
        n_events = max(1, int(np.random.poisson(lam=1.2*s*2)))
        for _ in range(n_events):
            ts = start + timedelta(days=random.randint(1, 35), hours=random.randint(0, 23))
            downtime = max(0.2, np.random.exponential(scale=1.5*s))  # hours
            cost = round(120.0 * downtime * (0.8 + 0.6*np.random.rand()), 2)
            rows.append({
                "event_ts": ts,
                "machine_id": m,
                "downtime_min": round(downtime * 60, 1),
                "failure_mode": random.choice(FAILURE_MODES),
                "repair_cost": cost
            })

    return pd.DataFrame(rows)

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate synthetic manufacturing datasets.")
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--parts", type=int, default=2000, help="Number of parts")
    ap.add_argument("--machines", type=int, default=15, help="Number of machines")
    ap.add_argument("--events-min", type=int, default=3, help="Min production events per part")
    ap.add_argument("--events-max", type=int, default=12, help="Max production events per part")
    ap.add_argument("--insp-min", type=int, default=1, help="Min inspections per part")
    ap.add_argument("--insp-max", type=int, default=3, help="Max inspections per part")
    ap.add_argument("--start", default="2025-01-10T08:00:00", help="Start timestamp (ISO)")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output format")
    # null injection controls (probabilities)
    ap.add_argument("--null-material", type=float, default=0.02, help="Prob. of NULL material")
    ap.add_argument("--null-sa", type=float, default=0.01, help="Prob. of NULL surface_area_mm2")
    ap.add_argument("--null-scrap-flag", type=float, default=0.03, help="Prob. of NULL scrap_flag")
    ap.add_argument("--null-energy", type=float, default=0.02, help="Prob. of NULL energy_kwh")
    ap.add_argument("--null-dimdev", type=float, default=0.05, help="Prob. of NULL dim_dev_um")
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Build datasets
    print(f"[+] Generating parts: {args.parts}")
    parts = make_parts(args.parts)

    avg_events = (args.events_min + args.events_max) / 2.0
    approx_prod_rows = int(args.parts * avg_events)
    print(f"[+] Generating production events (~{approx_prod_rows} rows)")
    prod = make_production(parts, (args.events_min, args.events_max), args.machines, args.start)

    print(f"[+] Generating quality inspections")
    qual = make_quality(parts, (args.insp_min, args.insp_max), args.start)

    print(f"[+] Generating maintenance events")
    maint = make_maintenance(prod, args.machines, args.start)

    # Inject NULLs for robustness testing
    parts = introduce_nulls(parts, {
        "material": args.null_material,
        "surface_area_mm2": args.null_sa,
    })
    prod = introduce_nulls(prod, {
        "scrap_flag": args.null_scrap_flag,
        "energy_kwh": args.null_energy,
    })
    qual = introduce_nulls(qual, {
        "dim_dev_um": args.null_dimdev,
    })
    # Optional: sparsely blank operator_id & scrap_reason too
    prod = introduce_nulls(prod, {"operator_id": 0.01, "scrap_reason": 0.01})

    # Save
    if args.format == "csv":
        parts.to_csv(os.path.join(args.outdir, "parts.csv"), index=False)
        prod.to_csv(os.path.join(args.outdir, "production.csv"), index=False)
        qual.to_csv(os.path.join(args.outdir, "quality.csv"), index=False)
        maint.to_csv(os.path.join(args.outdir, "maintenance.csv"), index=False)
    else:
        parts.to_parquet(os.path.join(args.outdir, "parts.parquet"), index=False)
        prod.to_parquet(os.path.join(args.outdir, "production.parquet"), index=False)
        qual.to_parquet(os.path.join(args.outdir, "quality.parquet"), index=False)
        maint.to_parquet(os.path.join(args.outdir, "maintenance.parquet"), index=False)

    print(f"[✓] Wrote datasets to: {args.outdir} ({args.format.upper()})")
    print("[i] NULL injection probs:",
          f"material={args.null_material}, sa={args.null_sa},",
          f"scrap_flag={args.null_scrap_flag}, energy={args.null_energy}, dim_dev={args.null_dimdev}")

if __name__ == "__main__":
    main()
