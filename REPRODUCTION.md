# Reproduction Guide

Written for someone starting from a clean machine with nothing installed.
No GPU, no API key, no database server, no network access after `pip install`.

---

## 0. What you need

| | |
|---|---|
| Python | 3.11 or newer (developed and measured on **3.14.4**) |
| Node | 20 or newer, **only** for the dashboard (developed on 22.21.0) |
| OS | Any. Developed on Windows 11; all paths are `pathlib`-based |
| Hardware | 4 CPU cores, ~4 GB RAM, ~600 MB disk. **No GPU** |
| Network | Needed once for `pip install` / `npm install`. Nothing else phones home |
| Credentials | **None.** No API key is required, and no result depends on one |

Total wall clock for the full pipeline: **≈ 25 minutes**, dominated by
training. Total cost: **$0.00** in the default configuration (see §7).

---

## 1. Install

```bash
git clone <repo> predictops
cd predictops

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` pins the versions the reported numbers were produced with.
PyTorch is the CPU build (~200 MB); the install is the slowest step, typically
2–5 minutes.

Verify:

```bash
python -c "import torch, xgboost, sklearn, pandas; print('ok')"
```

---

## 2. Generate the dataset

```bash
python generate_data.py --machines 80 --days 30 --seed 42
```

Runtime **≈ 5 s**. Writes to `artifacts/data/`:

| File | Contents |
|---|---|
| `telemetry.parquet` | 345,600 rows × 80 machines × 10-min resolution |
| `failures.parquet` | the failure events, with type and degradation start |
| `maintenance.parquet` | corrective and preventive service events |
| `machines.parquet` | machine metadata |
| `manifest.json` | full config + SHA-256 of every file |

Expected console output (this is the check that you reproduced the data):

```
"n_rows": 345600,
"n_machines": 80,
"n_failures": 112,
"positive_rate": 0.011666666666666667,
"downtime_rate": 0.009811921296296296,
"checksums": { "telemetry": "..." }
```

**The data is fully synthetic.** No real plant data, no personal data, no
licensed data. `--seed 42` reproduces it bit-for-bit
(`tests/test_generator.py::test_same_seed_is_bit_identical`).

---

## 3. Baseline

```bash
python train_baseline.py
```

Runtime **≈ 15 s** (first run also builds the cached feature frame, ~20 s).
Writes `artifacts/reports/baseline_test.json`.

Expected:

```
row.f1                                0.2581
row.precision                         0.2248
row.recall                            0.3030
row.pr_auc                            0.1518
event.detection_rate                  0.7308
event.false_alarms_per_machine_day    1.760
```

**Why this F1 is 0.2581 while the comparison tables say 0.2570.** This script
scores every usable test row; the model comparison scores the *canonical*
evaluation rows only — the rows that have a full clean lookback, so that a
sequence model and a tree are graded on identical cases
(`predictops/evaluation/harness.py`). Both figures are correct for what they
measure, and the reported comparison always uses the canonical set.

---

## 4. Model research (the main run)

```bash
python run_experiments.py            # ≈ 12 min on 4 CPU cores
python run_experiments.py --quick    # ≈ 6 min, fewer epochs (smoke test only)
```

This runs the Data Scientist and Model Research agents, trains seven
candidates, records every one in SQLite, selects a winner on **validation**
PR-AUC, assembles the deployable bundle, and writes the changelog.

Outputs:

| Path | Contents |
|---|---|
| `artifacts/experiments/experiments.db` | every experiment + every agent step |
| `artifacts/models/bundle/` | the deployable model bundle |
| `artifacts/reports/IMPROVEMENT_CHANGELOG.md` | generated from the registry |
| `artifacts/reports/experiments_<run-id>.json` | full run record |

Expected final table (reproduces exactly — the run is seeded end to end):

```
STAGE        MODEL                              VAL PR-AUC  TEST F1  DECISION
Baseline     threshold rule                         0.1446   0.2570  reference
Iteration 1  xgboost on raw features                0.3099   0.4430  kept
Iteration 2  xgboost on engineered features         0.5072   0.5118  kept
Iteration 3  LSTM on raw channels                   0.3640   0.4313  removed
Iteration 4  LSTM on engineered channels            0.5542   0.4937  kept
Iteration 5  TFT on engineered channels             0.5583   0.4659  removed
Iteration 6  ensemble lstm_engineered + tft_engin   0.5855   0.5142  kept

SELECTED: ensemble lstm_engineered + tft_engineered
  selected on validation PR-AUC 0.5855 (+0.0272 over TFT on engineered channels)
```

If your PR-AUC differs in the 4th decimal, that is BLAS thread-count variation
in the torch runs; the selection and the ordering are stable.

---

## 5. Evaluation — baseline vs agent

```bash
python evaluate.py                 # ≈ 35 s, full agent workflow on 45 cases
python evaluate.py --model-only    # ≈ 15 s, prediction only
```

Builds (or loads) the 45-scenario suite, runs the threshold baseline and the
agent workflow over identical cases, and writes
`artifacts/reports/evaluation.json`.

Prints the comparison table, the per-category accuracy breakdown, and the
capabilities the baseline does not have. Expected:

```
METRIC                                 SIMPLE BASELINE  AGENT SOLUTION       CHANGE
Alert accuracy (primary)                         68.9%           71.1%      +2.2 pp
F1                                               56.2%           62.9%      +6.6 pp
Precision                                       100.0%           91.7%      -8.3 pp
Recall                                           39.1%           47.8%      +8.7 pp
Cause accuracy (all real failures)               30.4%           39.1%      +8.7 pp
Cause accuracy (of those alerted)                77.8%           81.8%      +4.0 pp
Hard-case accuracy                               64.3%           67.9%      +3.6 pp
False alarms on nuisance cases                    0.0%            4.5%      +4.5 pp
```

The precision and nuisance rows are a **single** scenario (S37, a pressure
transducer spike): 1 false positive out of 22 nuisance cases against the
baseline's 0. That difference is below what a 45-case suite can resolve, and
the README says so rather than reporting it as a regression.

To rebuild the scenario suite from scratch:

```bash
python evaluate.py --rebuild-suite
```

The suite is drawn by rule from a fixed seed, so it is identical on every
machine (`test_scenario_suite_is_deterministic_and_hard`).

---

## 5b. Does the hypothesis contest earn its place?

```bash
python ablate_adjudication.py        # ≈ 30 s
```

Sweeps the alert threshold and compares model-only against adjudicated on the
same 45 scenarios. Expected output: **0 verdicts changed, +0.0000 F1**. That
negative result is deliberate and is discussed in the README.

---

## 6. One incident, end to end

```bash
python run_pipeline.py --scenario S01        # a specific evaluation case
python run_pipeline.py --machine PUMP-020    # a specific machine
python run_pipeline.py                       # the riskiest machine in the test period
python run_pipeline.py --scenario S01 --json # raw structured output
```

Runtime **≈ 1–2 s**. Prints the incident report and writes it, together with
the nine agent trajectory rows (prediction, context, investigation, both
hypothesis advocates, adjudication, remediation, simulation, verification).

---

## 7. LLM provider (optional)

Everything above runs with **no API key**. `get_provider()` resolves to
`MockProvider`, which returns each agent's deterministic findings unchanged.

To enable narrative generation:

```bash
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
export PREDICTOPS_LLM_PROVIDER=anthropic   # or openai, or mock to force off
python run_pipeline.py --scenario S01
```

**Cost.** The default is $0.00 — no tokens are sent. With Anthropic enabled,
an incident makes **4 structured calls** (investigation, both hypothesis
advocates, remediation), plus 2 once per research run (data scientist, model
research). Each is roughly 2–4k input and 300–600 output tokens. At Claude
Opus 5 rates that is well under **$0.15 per incident**; the full 45-case
evaluation stays under **$7**. Per-run spend is metered in `llm_usage` in
every report.

**What changes with an LLM, and what does not.** Only prose changes: the
factual summary, each advocate's argument and its stated refuter, the plan
summary. The advocates' *scores* and the adjudicator's decision are computed,
not written. Every number — risk,
window, failure type, confidence, simulated outcomes, verification verdicts —
is computed by fitted models and deterministic tools before the LLM is
consulted, and the verifier re-derives evidence claims from raw telemetry
either way. **The measured results in the README are identical with and
without an API key.**

---

## 8. Dashboard

```bash
cd frontend
npm install          # ≈ 1 min
npm run build        # ≈ 10 s
cd ..
uvicorn predictops.api.app:app --port 8000
```

Open **http://127.0.0.1:8000**. The API serves the built dashboard from the
same process, so one command gives you the whole product.

For frontend development with hot reload, run the API on 8000 and
`npm run dev` in `frontend/` (Vite proxies `/api` and `/ws` to 8000).

Pages: Fleet Command Center · Assistant · Machine Investigation ·
Remediation Simulator · Workflow Canvas ·
Agent Activity (live WebSocket stream) · Model Lab · Experiments · Evaluation.

---

## 9. Tests

```bash
python -m pytest              # 134 tests, ~9 min
python -m pytest -q tests/test_generator.py   # data integrity + leakage
python -m pytest -q tests/test_features.py    # causality + splits
python -m pytest -q tests/test_agents.py      # agent contracts + verification
python -m pytest -q tests/test_pipeline.py    # end to end
```

The tests worth knowing about:

| Test | What it protects |
|---|---|
| `test_features_do_not_depend_on_the_future` | recomputes every feature from a truncated history; catches any backfill or centred window |
| `test_positive_label_implies_failure_inside_horizon` | label correctness against the event table |
| `test_task_is_not_trivially_threshold_separable` | guards the premise — fails if one raw channel could solve it |
| `test_verifier_catches_fabricated_evidence` | plants a false number; asserts the run is marked FAIL |
| `test_pipeline_runs_without_any_api_key` | the no-credentials path |
| `test_bundle_round_trips` | the saved model scores identically after reload |

---

## 10. Full clean-environment sequence

```bash
git clone <repo> predictops && cd predictops
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py --seed 42     # 5 s
python train_baseline.py              # 15 s   -> F1 0.2581
python run_experiments.py             # 12 min -> selects the LSTM+TFT ensemble
python evaluate.py                    # 35 s   -> baseline vs agent
python -m pytest                      # 45 s   -> 66 passed

cd frontend && npm install && npm run build && cd ..
uvicorn predictops.api.app:app --port 8000
```

To start completely fresh, delete `artifacts/` — it is entirely regenerable.

---

## 11. Troubleshooting

**API says `setup_required` / routes return 503** — a clone carries no
generated dataset (`artifacts/data/` is regenerable, so it is gitignored). The
server starts anyway and `/api/health` names the fix; run `generate_data.py`,
then `run_experiments.py`, then restart it. The Experiments, Evaluation and
Agent Activity views work before that, because they read committed reports.

**`no model bundle; run run_experiments.py`** — `run_pipeline.py`, `evaluate.py`
and the API all need `artifacts/models/bundle/`. Run `run_experiments.py` first.

**Dashboard is a 404 at `/`** — the API serves `frontend/dist/`, which is a
build product and not committed. Run `cd frontend && npm install && npm run
build` once.

**Training is much slower than 12 min** — check nothing else is saturating the
CPU; the TFT is the long pole at ~45 s/epoch. Use `--quick` to smoke-test the
wiring.

**Stale features after regenerating data** — `prepare()` invalidates its cache
on the telemetry checksum automatically. To force it:
`python -c "from predictops.ml.dataset import prepare; prepare(force=True)"`.

**Dashboard shows "Could not load"** — the API is not running, or it started
before the bundle existed. Restart `uvicorn` after `run_experiments.py`.

**Evaluation page is empty** — run `python evaluate.py`, then reload.
