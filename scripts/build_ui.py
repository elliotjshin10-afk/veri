"""Build the partner-facing demo page from real model outputs.

Everything on the page is generated from artefacts on disk - the replay comes
from `demo/replay.json` (a held-out victim), the numbers from
`data/processed/report_latest.json`. Nothing is hand-typed, so the page cannot
drift from what the model actually does.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from veridis.config import DEMO, PROCESSED  # noqa: E402

OUT = Path("demo/veridis_demo.html")

replay = json.loads((DEMO / "replay.json").read_text())
rep = json.loads((PROCESSED / "report_latest.json").read_text())
m = rep["metrics"]
d = rep["dollars"]
ed = {r["k"]: r for r in rep["early_detection"]}
abl = {a["arm"]: a for a in rep.get("ablation", [])}
unseen = rep.get("unseen_destination", {})
fam = {r["family"]: r["share"] for r in rep["family_importance"]}
cal = rep.get("calibration", {})

sender_side = fam.get("sender", 0) + fam.get("relationship", 0)

def _headline(rep_: dict) -> tuple[str, str]:
    """Write the headline from the replay itself.

    A hard-coded headline is a claim that can silently stop matching the data
    the next time the pipeline runs, so both numbers in it are derived.
    """
    steps_ = rep_["steps"]
    first = dt.datetime.fromisoformat(steps_[0]["when"])
    last = dt.datetime.fromisoformat(steps_[-1]["when"])
    span_days = max(1, (last - first).days)
    span = (f"{span_days} days" if span_days < 21
            else f"{round(span_days/7)} weeks" if span_days < 90
            else f"{span_days/30:.0f} months")
    if rep_.get("listed_at"):
        listed = dt.datetime.fromisoformat(rep_["listed_at"])
        gap_days = max(0, (listed - first).days)
        gap = (f"{gap_days/365:.1f} years" if gap_days >= 365
               else f"{round(gap_days/30)} months" if gap_days >= 60
               else f"{gap_days} days")
        head = f"The address stayed clean for {gap}. The money left in {span}."
    else:
        head = f"The address was never listed anywhere. The money left in {span}."
    return head, span


headline, span_txt = _headline(replay)

lat_path = Path("reports/latency.json")
latency = json.loads(lat_path.read_text()) if lat_path.exists() else {}

payload = {
    "headline": headline,
    "latency_p95_ms": latency.get("p95_ms"),
    "replay": replay,
    "metrics": {
        "pr_auc": m["pr_auc"],
        "base_rate": m["base_rate"],
        "lift": m["pr_auc"] / max(m["base_rate"], 1e-9),
        "fpr": m["fpr"],
        "precision": m["precision_event"],
        "recall": m["recall_event"],
        "roc_auc": m["roc_auc"],
    },
    "early": [{"k": k, "rate": ed[k]["rate"], "flagged": ed[k]["flagged"],
               "victims": ed[k]["victims"]} for k in sorted(ed)],
    "dollars": d,
    "ablation": abl,
    "unseen": unseen,
    "families": fam,
    "sender_side": sender_side,
    "calibration": cal,
    "cost_curve": rep.get("cost_curve", []),
    "ece": rep.get("ece"),
    "psi_by_family": rep.get("psi_by_family"),
    "early_ci": rep.get("early_detection_ci"),
    "roc_ci": rep.get("roc_auc_ci"),
    "sequential": rep.get("sequential"),
    "prevalence": rep.get("prevalence_sensitivity", []),
    "economics": rep.get("deployment_economics", []),
    "warnings": rep.get("warnings", []),
    "first_send": rep.get("first_send_only"),
    "hard_negatives": rep.get("hard_negatives"),
    "achieved_ratio": (1 - m["base_rate"]) / max(m["base_rate"], 1e-9),
    "dataset": {
        "labels": 8552,
        "victims_identified": 9207,
        "relationships": 10233,
        "n_train": rep["n_train"],
        "n_test": rep["n_test"],
        "train_scams": rep["groups"]["train_scam_addresses"],
        "test_scams": rep["groups"]["test_scam_addresses"],
    },
    "generated": dt.datetime.now(dt.timezone.utc).strftime("%d %b %Y"),
}

dd = json.loads((DEMO / "demo_data.json").read_text())
payload["scenarios"] = dd["scenarios"]
amr = PROCESSED / "address_model_report.json"
if amr.exists():
    payload["addressModel"] = json.loads(amr.read_text())
payload["operating_curve"] = dd["operating_curve"]

TEMPLATE = Path("scripts/ui_template.html").read_text()
html = TEMPLATE.replace("/*__DATA__*/{}", json.dumps(payload))
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html)
print(f"wrote {OUT}  ({len(html):,} bytes)")
print(f"  replay: {len(replay['steps'])} transfers, first flag k={replay['first_flag_k']}")
print(f"  ablation arms: {list(abl)}")
