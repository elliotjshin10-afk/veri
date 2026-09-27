"""M0 — build the corroborated scam-address label set."""
import logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.labels.collect import build_reports, aggregate, finalise, write
from veridis.config import INTERIM

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

reports = build_reports()
reports.write_parquet(INTERIM / "scam_reports.parquet")
print(f"\nraw reports: {reports.height}")
print(reports.group_by(["source", "chain"]).len().sort("len", descending=True))

agg = aggregate(reports)
final = finalise(agg)
write(final)

print("\nby chain / accepted:")
print(final.group_by(["chain", "accepted"]).len().sort(["chain", "accepted"]))
print("\nlabel basis:")
print(final.group_by("label_basis").len().sort("len", descending=True))
tron = final.filter((pl.col("chain") == "tron") & pl.col("accepted"))
dated = tron.filter(pl.col("first_reported_at").is_not_null())
print(f"\ntron accepted={tron.height} with_timestamp={dated.height}")
if dated.height:
    print(dated.select(
        pl.from_epoch("first_reported_at", time_unit="ms").dt.year().alias("yr")
    ).group_by("yr").len().sort("yr"))
