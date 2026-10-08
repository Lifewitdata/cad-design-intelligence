"""
Generate synthetic CAD design-library datasets for the
'CAD Design Intelligence' portfolio project.

Outputs (in ../data/):
  cad_parts.csv          ~3,000 parts with CAD metadata
  design_changes.csv     ~8,000 change-history records
  manufacturing_rework.csv ~1,200 rework log rows

Built-in data-quality issues (for the notebook's audit step):
  - ~2% missing material, ~5% missing tolerance_class in cad_parts
  - 12 duplicate part_id rows in cad_parts
  - ~1.5% orphan part_ids in design_changes (not present in cad_parts)
  - a handful of impossible dates (change before part creation,
    last_modified before created)
  - a few negative / zero mass values

True underlying signals (for the analysis to discover):
  - tight tolerance  -> higher rework probability
  - high revision count -> higher rework probability
  - certain materials (abs_plastic) slightly worse
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

N_PARTS = 3000

# ---------------- 1. cad_parts ----------------
part_ids = [f"P-{i:04d}" for i in range(1, N_PARTS + 1)]
categories = ["bracket", "housing", "gear", "shaft",
              "connector", "pcb_mount", "fastener", "seal"]
materials = ["aluminum_6061", "steel_304", "abs_plastic",
             "nylon_66", "copper_c110", "titanium_6al4v"]
teams = ["mech_a", "mech_b", "elec_a", "hw_integration"]

cat = RNG.choice(categories, N_PARTS, p=[.18, .16, .12, .12, .12, .12, .10, .08])
mat = RNG.choice(materials, N_PARTS, p=[.30, .25, .18, .12, .08, .07])
tol = RNG.choice(["loose", "standard", "tight"], N_PARTS, p=[.30, .50, .20])
team = RNG.choice(teams, N_PARTS)
revs = RNG.poisson(2.5, N_PARTS) + 1
nfeat = RNG.poisson(22, N_PARTS) + 5

base_mass = {"bracket": .8, "housing": 2.5, "gear": 1.2, "shaft": 1.8,
             "connector": .15, "pcb_mount": .3, "fastener": .05, "seal": .1}
mass = np.array([RNG.lognormal(np.log(base_mass[c]), .5) for c in cat]).round(3)

created = pd.to_datetime("2022-01-01") + pd.to_timedelta(
    RNG.integers(0, 1276, N_PARTS), unit="D")
modified = created + pd.to_timedelta(RNG.integers(0, 400, N_PARTS), unit="D")

parts = pd.DataFrame({
    "part_id": part_ids,
    "part_category": cat,
    "material": mat,
    "mass_kg": mass,
    "tolerance_class": tol,
    "revision_count": revs,
    "num_features": nfeat,
    "designer_team": team,
    "created_date": created.date.astype(str),
    "last_modified_date": modified.date.astype(str),
})

# --- inject DQ issues ---
parts.loc[RNG.choice(parts.index, 60, replace=False), "material"] = np.nan          # 2% missing
parts.loc[RNG.choice(parts.index, 150, replace=False), "tolerance_class"] = np.nan  # 5% missing
dups = parts.sample(12, random_state=7).copy()                                      # duplicate part_ids
dups["revision_count"] = dups["revision_count"] + 1                                 # ...with conflicting data
parts = pd.concat([parts, dups], ignore_index=True)
bad_mass = RNG.choice(parts.index, 8, replace=False)
parts.loc[bad_mass, "mass_kg"] = np.round(RNG.uniform(-2, 0, 8), 3)                 # impossible mass
bad_dt = RNG.choice(parts.index, 10, replace=False)
parts.loc[bad_dt, "last_modified_date"] = (                                        # modified before created
    pd.to_datetime(parts.loc[bad_dt, "created_date"]) - pd.to_timedelta(30, unit="D")
).dt.date.astype(str)

# ---------------- 2. design_changes ----------------
change_types = ["dimension_update", "material_change", "tolerance_tightening",
                "feature_add", "feature_remove", "drawing_note"]
reasons = ["design_review", "test_failure", "supplier_issue",
           "cost_reduction", "customer_request"]

rows = []
chg = 0
for _, p in parts.drop_duplicates("part_id").iterrows():
    n_ch = min(int(p["revision_count"]) + int(RNG.integers(0, 3)), 12)
    start = pd.to_datetime(p["created_date"])
    for r in range(1, n_ch + 1):
        chg += 1
        rows.append({
            "change_id": f"CHG-{chg:05d}",
            "part_id": p["part_id"],
            "revision_number": r,
            "change_type": RNG.choice(change_types),
            "reason_code": RNG.choice(reasons, p=[.30, .25, .15, .15, .15]),
            "changed_by_team": p["designer_team"],
            "change_date": (start + pd.to_timedelta(int(RNG.integers(0, 500)), unit="D")).date().isoformat(),
        })
changes = pd.DataFrame(rows)

# orphan part_ids (~1.5%)
orph = pd.DataFrame({
    "change_id": [f"CHG-{chg + i + 1:05d}" for i in range(120)],
    "part_id": [f"P-{RNG.integers(9000, 9999)}" for _ in range(120)],
    "revision_number": RNG.integers(1, 4, 120),
    "change_type": RNG.choice(change_types, 120),
    "reason_code": RNG.choice(reasons, 120),
    "changed_by_team": RNG.choice(teams, 120),
    "change_date": "2024-06-15",
})
changes = pd.concat([changes, orph], ignore_index=True)
# impossible dates: a few changes before the part was created
imp = changes.sample(15, random_state=11).copy()
imp["change_date"] = "2021-03-10"
changes = pd.concat([changes, imp], ignore_index=True)

# ---------------- 3. manufacturing_rework ----------------
# risk score drives which parts get reworked (the discoverable signal)
base = parts.drop_duplicates("part_id").copy()
risk = (
    0.35 * (base["revision_count"] - 1)
    + 1.4 * (base["tolerance_class"] == "tight").astype(float)
    + 0.5 * (base["material"] == "abs_plastic").astype(float)
    + 0.02 * base["num_features"]
    - 2.2
)
prob = 1 / (1 + np.exp(-risk))
prob = np.nan_to_num(prob, nan=np.nanmean(prob))
prob = prob / prob.sum()

n_rework = 1200
picked = RNG.choice(base["part_id"], n_rework, replace=True, p=prob)
rework_types = ["rework_machining", "scrap_replace", "assembly_adjust", "retest_only"]
root_causes = ["tolerance_stackup", "design_error", "material_defect",
               "process_variation", "supplier_quality"]

rework = pd.DataFrame({
    "rework_id": [f"RWK-{i:05d}" for i in range(1, n_rework + 1)],
    "part_id": picked,
    "rework_type": RNG.choice(rework_types, n_rework, p=[.35, .25, .25, .15]),
    "rework_cost_usd": np.round(RNG.lognormal(4.5, 0.9, n_rework), 2),
    "root_cause": RNG.choice(root_causes, n_rework, p=[.30, .25, .15, .20, .10]),
    "rework_date": (pd.to_datetime("2023-01-01") + pd.to_timedelta(
        RNG.integers(0, 910, n_rework), unit="D")).date.astype(str),
})

# ---------------- write ----------------
parts.to_csv(OUT / "cad_parts.csv", index=False)
changes.to_csv(OUT / "design_changes.csv", index=False)
rework.to_csv(OUT / "manufacturing_rework.csv", index=False)

print("cad_parts.csv           :", parts.shape)
print("design_changes.csv      :", changes.shape)
print("manufacturing_rework.csv:", rework.shape)
print("rework rate (unique parts reworked / unique parts):",
      round(rework["part_id"].nunique() / base["part_id"].nunique(), 3))
