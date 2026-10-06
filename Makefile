# Use the local venv when there is one, otherwise whatever python is on PATH.
# The hardcoded venv path meant every target failed on a clean checkout, which
# is exactly what CI is.
PY := $(shell [ -x ./.venv/bin/python ] && echo ./.venv/bin/python || echo python3)
export DYLD_LIBRARY_PATH := /opt/homebrew/opt/libomp/lib:$(DYLD_LIBRARY_PATH)

.PHONY: freezes history setup labels scam-history victims controls harvest features train eval crosschain api demo bench ui research browser-model parity site test dataset clean

setup:
	/opt/homebrew/bin/python3.11 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt
	@echo "macOS: LightGBM needs OpenMP -> brew install libomp"

## ---- dataset (M0-M2) -------------------------------------------------
labels:
	$(PY) scripts/m0_labels.py

scam-history:
	$(PY) -u scripts/m1_scam_histories.py $(N) $(PAGES)

victims:
	$(PY) -u scripts/m1b_victims.py $(N) $(PAGES)

controls:
	$(PY) -u scripts/m2_controls.py

## Rebuild dataset tables from the HTTP cache alone, making no requests.
## Use after a quota-limited ingest stalls part-way.
N ?= 200
PAGES ?= 3
harvest:
	$(PY) -u scripts/harvest_cache.py victims $(PAGES)
	$(PY) -u scripts/harvest_cache.py controls $(PAGES)

dataset: labels scam-history victims controls features

## ---- model (M3-M4) ---------------------------------------------------
features:
	$(PY) -u scripts/m3_features.py

train:
	$(PY) -u scripts/m4_train.py

eval:
	$(PY) -u scripts/m4_train.py

crosschain:
	$(PY) -u scripts/m7_crosschain.py

## ---- product (M5-M6) -------------------------------------------------
api:
	$(PY) -m uvicorn veridis.api.main:app --app-dir src --host 127.0.0.1 --port 8000

browser-model:
	$(PY) -u scripts/build_browser_model.py

parity:
	$(PY) -u scripts/check_parity.py $(N)

site:
	$(PY) -u scripts/build_site.py

## The decided-cases log: held-out transfers, the verdict each got before it
## settled, and what the chain did next. Rebuild it whenever model_pair.json is
## retrained, or the page cites a model it no longer ships.
history:
	$(PY) -u scripts/build_history.py

# Only the five files the page fetches at runtime. Deploy dist/, not site/.
dist: site
	$(PY) -u scripts/build_dist.py

ui:
	$(PY) -u scripts/build_ui.py
	@echo "open demo/veridis_demo.html"

research:
	$(PY) -u scripts/research.py $(ADDR)

bench:
	$(PY) -u scripts/bench_api.py

report:
	$(PY) -u scripts/make_report.py

demo:
	$(PY) -u scripts/m6_demo.py
	$(PY) -u scripts/render_demo.py
	@echo "open demo/index.html"

## ---- gates -----------------------------------------------------------
test:
	$(PY) -m pytest tests/ -q
	@echo '--- browser/Python scoring parity ---'
	$(PY) -u scripts/check_parity.py 200
	@echo '--- browser/Python reason-text parity ---'
	$(PY) -u scripts/check_notes_parity.py
	@echo '--- two-sided (sender+destination) feature parity ---'
	$(PY) -u scripts/check_pair_parity.py 300
	@echo '--- Etherscan normaliser parity (Ethereum) ---'
	$(PY) -u scripts/check_evm_parity.py 200
	@echo '--- Etherscan refusal handling (browser) ---'
	node scripts/check_js_refusal.mjs
	@echo '--- browser/LightGBM parity, Ethereum models ---'
	$(PY) -u scripts/check_eth_parity.py

clean:
	rm -rf data/interim/* data/processed/* reports/*

# --- prediction ledger -------------------------------------------------
# Nightly: flag unlisted addresses, then check earlier calls against the
# current freeze list. Order matters - resolve before predict would leave the
# newest entries unchecked for a day.
# ~1% of candidates clear the 1% threshold, so the candidate count sets how
# fast the ledger accrues: 2,000 a night is roughly 20 entries, ~10 minutes of
# fetching, and about 1,800 calls on the record by the end of a quarter.
N ?= 2000
# The leading `-` on the refresh is deliberate. Resolution needs a freeze list
# newer than the predictions it checks, and for nine nights it had one fetched
# before the ledger even started - so every run reported "0 resolved" and that
# read as evidence rather than as a broken loop. But a third-party outage must
# not also cost a night of PREDICTIONS, which is the half that only accrues with
# calendar time. So a failed refresh is tolerated here and caught instead by
# tests/test_ledger_resolve.py, which fails once the lists stop reaching past
# the ledger: a transient miss is absorbed, a persistent one is loud.
ledger:
	-$(PY) -u scripts/refresh_freezes.py
	$(PY) -u scripts/ledger_predict.py $(N)
	$(PY) -u scripts/ledger_resolve.py
	$(PY) -u scripts/ledger_report.py

## Bring the Tether freeze lists up to date on their own (both chains).
freezes:
	$(PY) -u scripts/refresh_freezes.py

ledger-verify:
	$(PY) -u scripts/ledger_verify.py
