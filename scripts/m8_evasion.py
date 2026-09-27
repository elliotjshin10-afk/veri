"""Run the adversarial stress test against the trained model."""
import json, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import polars as pl

from veridis.features.asof import FEATURE_COLUMNS
from veridis.model.evasion import ATTACKS, evaluate_attack
from veridis.model.train import temporal_split
from veridis.config import PROCESSED

warehouse = pl.read_parquet(PROCESSED / "warehouse_transfers.parquet")
events = pl.read_parquet(PROCESSED / "events_features.parquet")
split = temporal_split(events)
booster = lgb.Booster(model_file=str(PROCESSED / "model_latest.txt"))
report = json.loads((PROCESSED / "report_latest.json").read_text())
threshold = report["metrics"]["threshold"]

# Attack only the held-out victims; controls are untouched, because a scammer
# cannot change how legitimate users behave.
test_pos = split.test.filter(pl.col("label") == 1)
pairs = (test_pos.select(pl.col("sender").alias("from_address"),
                         pl.col("destination").alias("to_address")).unique())
controls = split.test.filter(pl.col("label") == 0).select(
    "chain", "sender", "destination", "amount_usd", "asset", "event_time",
    "tx_hash", "transfer_k", "label")
print(f"attacking {pairs.height} held-out victim relationships; "
      f"{controls.height:,} controls untouched")
print(f"threshold held fixed at its pre-attack value ({threshold:.4f})\n")

rows = []
for name in ATTACKS:
    r = evaluate_attack(warehouse, pairs, controls, booster, threshold,
                        name, FEATURE_COLUMNS)
    rows.append(r)
    print(f"  {name:<26} detected {r['detected_ever']:>6.1%}  "
          f"by k=3 {r['detected_by_k3']:>6.1%}  "
          f"dollars {r['dollars_share']:>6.1%}  "
          f"(FPR {r['control_fpr']:.2%})")

# The decisive comparison: which half of the signal survives the attack that
# actually threatens it. A recipient-side system has only the destination arm.
from veridis.model.ablation import ARMS, _cols
from veridis.model.evaluate import threshold_at_fpr
import lightgbm as _lgb
import numpy as _np

arm_rows = []
for arm in ("destination_only", "sender_side_only", "full"):
    acols = _cols(ARMS[arm])
    ytr = split.train["label"].to_numpy()
    ds = _lgb.Dataset(split.train.select(acols).to_numpy(), label=ytr,
                      feature_name=acols)
    ab = _lgb.train({"objective": "binary", "learning_rate": 0.05,
                     "num_leaves": 31, "min_data_in_leaf": 30,
                     "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1),
                     "verbose": -1, "seed": 17, "num_threads": 4},
                    ds, num_boost_round=300)
    # Each arm gets its own threshold at the same budget, pre-attack.
    s_te = ab.predict(split.test.select(acols).to_numpy())
    athr = threshold_at_fpr(split.test["label"].to_numpy(), s_te, 0.01)
    for name in ("none", "fresh_address_per_victim", "all_combined"):
        r = evaluate_attack(warehouse, pairs, controls, ab, athr, name, acols)
        arm_rows.append({"arm": arm, **r})

print(f"\n{'arm':<20}{'no attack':>12}{'fresh address':>16}{'all combined':>15}")
for arm in ("destination_only", "sender_side_only", "full"):
    got = {r["attack"]: r["detected_ever"] for r in arm_rows if r["arm"] == arm}
    print(f"{arm:<20}{got.get('none',0):>12.1%}"
          f"{got.get('fresh_address_per_victim',0):>16.1%}"
          f"{got.get('all_combined',0):>15.1%}")

base = rows[0]
print(f"\n{'attack':<26}{'detection':>12}{'vs baseline':>14}{'dollars':>10}{'vs base':>10}")
for r in rows:
    d = r["detected_ever"] - base["detected_ever"]
    m = r["dollars_share"] - base["dollars_share"]
    print(f"{r['attack']:<26}{r['detected_ever']:>12.1%}{d:>+14.1%}"
          f"{r['dollars_share']:>10.1%}{m:>+10.1%}")

(PROCESSED / "evasion_report.json").write_text(
    json.dumps({"attacks": rows, "by_arm": arm_rows}, indent=2))
print(f"\nwrote evasion_report.json")
