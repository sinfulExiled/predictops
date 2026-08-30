# PredictOps — Agentic Predictive Failure Intelligence

**PredictOps turns a failure prediction into a defensible maintenance
decision.** A trained temporal model answers whether a machine is likely to
fail; a team of agents then works out what is actually known, what could
explain it, what the evidence supports, what to do, and whether there is enough
of a case to act at all.

It does not claim that agents make the prediction more accurate. We measured
that, and they do not — see §6. What they do is turn a number into something a
maintenance planner can defend to their supervisor.

| Layer | Question it answers |
|---|---|
| **ML** | Is failure likely, when, and of what kind? |
| **Agents** | What do we know? What could explain it? What does the evidence support? What should we do? |
| **Simulation** | What might happen if we do it? |
| **Verification** | Do we have enough evidence to act? |

**The prediction comes from a trained ML model, never from a language model.**
An LLM, when one is configured, ranks hypotheses and writes prose over evidence
the tools already computed — it is never asked for a number, and every measured
result in this README is identical with and without an API key.

---

## 1. The problem, and who has it

**Who.** The reliability engineer at a mid-sized **water and wastewater
treatment works** — one plant, 80 rotating assets, a two-person maintenance
crew, and a historian that has been logging sensor data every ten minutes for
years without anyone having time to look at it.

The fleet is what such a site actually runs, and it is pump-dominated:

| Class | Count | What they are |
|---|---|---|
| Pumps | 32 | raw-water, transfer, return-activated-sludge and dosing |
| Compressors | 20 | aeration blowers, plus the instrument-air package |
| Motors | 20 | mixer, screen and clarifier drives |
| Conveyors | 8 | screenings and dewatered-sludge handling |

**The bottleneck.** They already have condition monitoring, and it is a
nuisance generator. Fixed alarm thresholds fire on anything unusual — and on
this plant, "unusual" is usually innocent: a storm-flow surge that ramps every
transfer pump, a hot afternoon, a fouled transducer. In this project's measured
baseline, a tuned per-machine threshold rule produces **2.06 false alarms per
machine per day**. Across 80 machines that is ~165 nuisance callouts a day for
two people. The rational response is to stop trusting the alarms — and then the
one that mattered is missed too.

The gap is not detection. It is **discrimination plus explanation**: telling a
cavitating transfer pump apart from a wet Tuesday, and handing the engineer
enough evidence to act on it before the machine stops.

**Why it is worth solving.** A failed raw-water or RAS pump is not an
inconvenience: it is lost treatment capacity, a possible consent breach, and a
conversation with the regulator. The difference between a planned four-hour
bearing change and an unplanned failure is roughly an order of magnitude in
cost, before any environmental penalty. But acting on a bad prediction is
expensive too — a two-person crew has no spare callouts — which is why this
system reports what it does *not* know as carefully as what it does.

**On the domain.** The failure physics modelled here is equipment-level —
bearing wear, cavitation, overheating, pressure loss, electrical faults — so
the same models would serve any rotating-equipment plant. Water treatment is
the framing because it fits the fleet mix, and because the cost of a missed
failure there is concrete enough to argue about.

---

## 2. What it does

```
telemetry ─► Data Scientist ─► Model Research ─►┐
                                                │   MODEL SERVICE
                                                └─► XGBoost / LSTM / TFT
                                                          │
   ┌──────────────────────────────────────────────────────┘
   ▼
Prediction ─► Context ─► Investigation ─┬─► Degradation advocate ─┐
 "what will   "what do    "what          │                        ├─► Adjudicator
  happen?"     we know?"   changed?"     └─► Confound advocate ───┘   "which reading
                                                                        survives?"
                                                          │
                    ┌─────────────────────────────────────┘
                    ▼
     Remediation ─► Simulation ─► Verification ─► human approval ─► report
     "what can we    "what would   "does any of
      do?"            that buy?"    this hold up?"
```

**The models are instruments, not agents.** `ml/service.py` is the boundary:
an agent asks a question, a fitted model answers. That is why the research
agent could swap XGBoost in for the TFT without touching a single downstream
agent.

**Each agent owns a different question**, which is the test for whether it
should exist at all:

| Agent | Question it owns |
|---|---|
| Data Scientist | Is the data trustworthy? |
| Model Research | Which model should we use? |
| Prediction | What is likely to happen? |
| Context | What do we already know about this machine? |
| Investigation | What actually changed? |
| Degradation advocate | Why is a fault the best explanation? |
| Confound advocate | Why is nothing wrong? |
| Adjudicator | Which reading survives? |
| Remediation | What can we do? |
| Simulation | What happens if we do it? |
| Verification | Do we actually have enough evidence? |
| Assistant | What is the operator asking, and can we ground it? |

Given one machine at one moment it answers six questions, and labels every
figure with its provenance — `measured`, `model`, or `simulated`:

| Question | Answered by | Source |
|---|---|---|
| **What?** will it fail | XGBoost / LSTM / TFT risk model | `model` |
| **When?** | time-to-failure regressor | `model` |
| **Why?** | evidence recomputed from raw telemetry + model attribution | `measured` |
| **What do we do?** | approved intervention catalogue | policy |
| **What does that buy?** | counterfactual rollout re-scored by the model | `simulated` |
| **How sure are we?** | measured tail precision + 9 verification checks | `measured` |

---

## 3. Quick start

Python 3.11+ (developed on 3.14), no GPU, no API key.

```bash
pip install -r requirements.txt

python generate_data.py --machines 80 --days 30 --seed 42   # ~5 s
python train_baseline.py                                     # ~15 s
python run_experiments.py                                    # ~20 min
python evaluate.py                                           # ~4 min
python run_pipeline.py --scenario S01                        # ~2 s
python ablate_adjudication.py                                # ~30 s
```

Then the dashboard:

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn predictops.api.app:app --port 8000
# open http://127.0.0.1:8000
```

**No API key is required for any of this**, and none of the measured results
depend on one — see §7.

---

## 4. Why the agents are built this way

Each design choice exists to kill a failure mode that showed up while
building, not because the component was on a list.

**Tools compute, the LLM narrates.** Every agent runs `investigate()` — pure
deterministic tool work — and only then an optional `narrate()` pass over the
result. The `MockProvider` returns the computed findings verbatim, so the
pipeline runs identically with no LLM at all. This is what makes the
evaluation numbers reproducible for someone with no API key.

**Evidence carries a recompute recipe.** Every evidence item ships the exact
function, channel and window that produced it:

```json
{"id": "E1", "claim": "Vibration rose 178% over the last 3 hours",
 "value": 178.34, "recompute": {"fn": "pct_change", "channel": "vibration", "hours": 3.0}}
```

The verifier re-runs that function against its **own** independent read of the
raw telemetry and compares. `tests/test_agents.py::test_verifier_catches_fabricated_evidence`
plants a false number and asserts the run is marked `FAIL`. That test is the
project's central claim in executable form.

**The remediation agent selects an id, it cannot write one.** Actions come
from `simulation/interventions.py`. An unknown id raises. High-risk or
downtime-causing actions are forced through an approval gate — verified by
check C9, and by a test asserting every high-risk catalogue entry is gated.

**The simulation has a control arm.** Every action is compared against
*doing nothing* under an identical rollout, so the delta isolates the
intervention rather than the passage of time.

**The assistant retrieves; it does not recall.** A chat surface is the easiest
place to undo everything else, so it works in three fixed steps: a
deterministic intent router, retrieval from computed artifacts with a citation
per number, then an optional LLM rephrasing that is **discarded if it
introduces a number not in the retrieved facts** — the same check the verifier
applies to the investigation narrative. It may run analysis (score a machine,
run the workflow, simulate an action) because those compute and change nothing
physical. It may never approve, schedule or execute work: the refusal filter is
checked before any retrieval, and 13 phrasings of it are tested. Questions it
cannot ground get "I don't have a computed answer for that", not a plausible
sentence.

**The workflow is data, and the canvas edits it.** `agents/workflow.py`
declares what each agent consumes and produces; the Workflow Canvas lets you
rewire and re-run the graph, and every rejection comes from those contracts
rather than a UI rule — *"Adjudicator needs degradation but no connected
upstream node provides it"*. Deleting an edge genuinely starves the target,
because a node only sees state produced by a connected ancestor. That is what
makes dropping the confound advocate a real ablation you can run in the
browser rather than a redrawn line.

**Model selection is arithmetic, not opinion.** `ModelResearchAgent.select()`
maximises validation PR-AUC and breaks near-ties toward the simpler model.
Test metrics are never consulted for selection. The LLM may explain the
choice; it cannot make it.

---

## 5. Results

Full generated changelog: [`artifacts/reports/IMPROVEMENT_CHANGELOG.md`](artifacts/reports/IMPROVEMENT_CHANGELOG.md)
(regenerated from the registry — no number in it is hand-entered).

### Model bake-off — identical data, identical evaluation rows

Every candidate is scored on the same canonical row set (`evaluation/harness.py`),
because sequence models can only score rows with a full clean lookback while
tabular models can score everything. Comparing them on different row sets would
have quietly rigged the result.

| Stage | Model | Val PR-AUC | Test F1 | Decision |
|---|---|---|---|---|
| Baseline | threshold rule | 0.1446 | 0.2570 | reference |
| Iteration 1 | XGBoost, raw features | 0.3099 | 0.4430 | kept |
| Iteration 2 | XGBoost, engineered features | 0.5072 | 0.5118 | kept |
| Iteration 3 | LSTM, raw channels | 0.3640 | 0.4313 | removed |
| Iteration 4 | LSTM, engineered channels | 0.5542 | 0.4937 | kept |
| Iteration 5 | TFT, engineered channels | 0.5583 | 0.4659 | removed |
| Iteration 6 | **Ensemble LSTM + TFT** | **0.5855** | **0.5142** | **selected** |

**Test F1 exactly doubled over the baseline: 0.257 → 0.514 (+100%).**

The single largest contribution is **feature engineering, not architecture**.
Moving from raw channels to causal rolling features is worth **+0.197**
validation PR-AUC to XGBoost and **+0.190** to the LSTM. The best architectural
change in the entire run — blending the two sequence models — is worth
**+0.027**, and swapping an LSTM for a TFT is worth **+0.004**. An order of
magnitude separates the two kinds of work.

### What the plant actually feels

Row-level PR-AUC is the selection metric. This is the table the maintenance
planner cares about, over the 26 real failure events in the test period:

| | Events caught | Mean early warning | **False alarms / machine / day** |
|---|---|---|---|
| Threshold baseline | 19 / 26 (73%) | 4.39 h | **2.06** |
| **PredictOps (selected: ensemble)** | 16 / 26 (62%) | 4.72 h | **0.46** |
| *XGBoost engineered (ran, not selected)* | *21 / 26 (81%)* | *4.51 h* | *0.69* |

Against the alarm rule the plant already runs, the deployed model cuts false
callouts **4.5×** — across 80 machines, ~165 nuisance callouts a day falling to
~37 — and adds about 20 minutes of warning. It also catches three fewer of the
26 events. Read the third row before accepting that trade.

### The result we did not expect

**The selection metric and the operational metric disagree, and we are shipping
the model the selection metric chose.**

The ensemble won on validation PR-AUC (0.5855, +0.027 over the TFT) and went on
to post the best test F1 of any candidate (0.5142). By the rule fixed before
the run, it is the winner. But the model it beat — XGBoost on engineered
features, val PR-AUC 0.5072 — catches **21 of 26 failure events to the
ensemble's 16**, for 0.69 false alarms per machine-day against 0.46.

Those two facts are not in conflict; they are measuring different things.
PR-AUC scores 83,837 ten-minute rows as if each were an independent question.
A planner is not asked 83,837 questions — they are asked 26, one per failure,
and a run of rows correctly flagged inside one long degradation counts once.
The ensemble is the more *precise* model per row; XGBoost is the more
*sensitive* model per event.

We did not switch. Changing the selection metric after seeing which model it
would favour is exactly the selection-on-the-outcome this project was built to
avoid, and the honest report is the disagreement itself: **a plant that cannot
afford to miss five failures in a quarter should deploy the XGBoost model, and
the selection rule as written would not have handed it to them.** The fix is
not a different winner, it is a different metric declared up front — event
recall at a fixed false-alarm budget — and that is a change for the next run,
not this one.

The brief warned against assuming TFT + LSTM would win. Here they did win — but
only because they were made to earn it against five alternatives on a metric
fixed in advance, and the same discipline is what exposed the metric's own
limits. Not assuming the architecture and not assuming the yardstick turn out
to be the same habit.

---

## 6. Evaluation

45 fixed scenarios drawn by rule from the held-out test period — 23 real
warning windows and 22 nuisance cases, 25 rated *hard*. Both systems see
identical cases and identical telemetry, and both have their threshold tuned
on the same validation split and frozen before the suite is touched.

Categories include the cases that separate a predictor from a threshold:
`subtle_degradation`, `sudden_failure`, `atypical_pattern` (the mode's
signature channel suppressed), `hot_weather_no_failure`,
`load_surge_no_failure`, `sensor_spike_no_failure`, `missing_telemetry`,
`multiple_anomalies_no_failure`.

| Metric | Simple baseline | Agent solution | Change |
|---|---|---|---|
| **Alert accuracy** (primary) | 68.9% | **71.1%** | +2.2 pp |
| F1 | 56.2% | **62.9%** | +6.6 pp |
| Precision | **100.0%** | 91.7% | −8.3 pp |
| Recall | 39.1% | **47.8%** | +8.7 pp |
| Cause accuracy — over *all* real failures | 30.4% | **39.1%** | +8.7 pp |
| Cause accuracy — over the ones it *alerted on* | 77.8% | **81.8%** | +4.0 pp |
| Hard-case accuracy | 64.3% | **67.9%** | +3.6 pp |
| **False alarms on nuisance cases** | **0.0%** | 4.5% | +4.5 pp |
| Seconds per case | 0.014 | 0.641 | — |

Confusion, since the percentages above hide how few cases each rests on
(23 real warning windows, 22 nuisance):

| | TP | FP | FN | TN |
|---|---|---|---|---|
| Baseline | 9 | 0 | 14 | 22 |
| Agent | 11 | 1 | 12 | 21 |

**The precision and nuisance rows are one case.** The agent fires on one of the
22 nuisance scenarios and the baseline fires on none; rendered as percentages
that reads as an 8.3-point precision regression, which is more than a 45-case
suite can resolve. What the suite *does* support is the recall column — the
agent finds two failures the baseline misses at the cost of that one callout —
and the row-level result behind it, measured over 83,837 test rows rather than
45: F1 0.257 → 0.514.

Cause accuracy is reported two ways because they answer different questions,
and only quoting the flattering one would be misleading. An earlier version of
this table showed a single 69.6% figure; that was **inflated** — it read the
investigator's internal ranking even on cases where the system never raised an
alert, crediting it for a diagnosis no human would ever have seen. Silence now
counts as naming no cause, which is what drops the coverage figure to 39.1%.

Cost per case: **$0.00** in the default configuration (no LLM call is required).

**The agent held on 21 of 22 nuisance cases** — hot weather, load surges,
sensor spikes, dropouts and simultaneous anomalies — against 22 of 22 for the
baseline, which achieves that by alerting on very little at all. Rejecting
confounders is most of what separates the two systems on cause accuracy.

**Where it is weak, stated plainly.** Recall is 47.8%: it misses more than half
the warning windows at this operating point. Specifically it catches **0 of 4
sudden failures** (degradation developing in under 2.5 hours — with a 6-hour
horizon and a 6-hour lookback there is often no pre-failure signal to find),
and only 50% of `early_warning` cases at the far edge of the horizon. The
system trades recall for precision. For this user that is the right trade — an
ignored alarm system has zero recall in practice — but it is a trade, not a
free win, and a plant that cannot tolerate missed events should move the
threshold and accept the callouts.

Run `python evaluate.py`; results land in `artifacts/reports/evaluation.json`
and render on the dashboard's Evaluation page.

**The baseline is given a cause-attribution rule too** (the tripped channel
mapped to the failure mode it most often signals), so *cause accuracy* is a
real comparison rather than a category it structurally cannot score in.
Capabilities the baseline genuinely does not have — verification verdicts,
simulated risk reduction — are **reported, not scored as zero**. Scoring a
system at zero for a question it was never asked is not a measurement.

### Does the hypothesis contest earn its place? — measured, and the answer is "not on accuracy"

The architectural claim was: because a flagged case must survive a benign
counter-argument, the model can afford a **lower** trigger and buy recall
without paying in false callouts. `ablate_adjudication.py` tests exactly that,
sweeping the alert threshold and comparing model-only against adjudicated on
the same 45 scenarios.

| Alert threshold | Model only (P / R / F1) | Adjudicated (P / R / F1) | Verdicts changed |
|---|---|---|---|
| 0.987 (tuned) | 0.917 / 0.478 / 0.629 | 0.917 / 0.478 / 0.629 | 0 |
| 0.740 | 0.867 / 0.565 / **0.684** | 0.867 / 0.565 / **0.684** | 0 |
| 0.247 | 0.765 / 0.565 / 0.650 | 0.765 / 0.565 / 0.650 | 0 |

**Adjudication is worth +0.0000 F1. It never changed a verdict.**

And the first version was *worse* than that: without a floor on the degradation
case, it changed four verdicts across the sweep and **all four were wrong**
(−0.039 F1). The failing case is instructive — S21, a genuine failure one hour
out, was overturned because the duty happened to rise at the same time, so the
confound advocate scored 0.87 against degradation's 0.56. The fix is a rule
worth stating: **a benign explanation may break a marginal case, never a strong
one** (`DEGRADATION_FLOOR` in `agents/hypothesis.py`).

Why is it inert? Not, any longer, for want of a target. The suite contains
exactly one false positive, and it is precisely the case the confound advocate
was built for — **S37, an isolated pressure-transducer glitch on PUMP-030 with
no underlying trend**. Here is what every component did with it:

| Component | Output on S37 |
|---|---|
| Ensemble model | **99.1%** failure probability (threshold 98.7%) |
| Degradation advocate | 0.88 |
| **Confound advocate** | **0.00 — it did not argue at all** |
| Adjudicator | alert, margin +0.88 |
| Verifier | **PASS on all ten checks**, including C5 "benign alternatives considered" and C10 "resolved on evidence" |
| Remediation | cleared to act: `restore_suction`, $350, medium risk, 0.5 h downtime |
| Reliability curve | **55%** — the only component that hedged |

A spike is a *confound*; recognising one is the confound advocate's entire
purpose; it scored the benign explanation at zero. The contest did not fail to
overturn a marginal call — it never contested. And C5 passed because it checks
that alternatives were *considered*, not that they were considered *well*: a
0.00 counter-argument is a considered alternative by that definition. So the
one case where the architecture should have earned its keep is also the case
where it, and the check meant to police it, were both silent together.

The reliability curve was right, at 55%. That number was on the screen and no
gate was reading it.

**So what is it kept for, honestly?** Not accuracy. It changes what the system
*does* and what the human *sees*:

- a `contested` verdict routes to **inspect**, never to **repair** — the
  remediation agent is gated on the adjudication, and a case that has not
  survived challenge cannot propose a bearing replacement or a shutdown
  (`test_an_overturned_case_proposes_no_physical_work`);
- the planner sees both readings and the margin, so "0.92 vs 0.00" and
  "0.56 vs 0.48" are visibly different situations rather than two identical
  87%s.

That is a real contribution to decision quality and a real one to safety. It is
not a contribution to F1, and this README does not claim one.

### A third baseline, built but not run

The brief also suggests "one direct prompt with basic instructions" as a
baseline, and this project's central architectural claim is that a language
model should not be the predictor. `llm_baseline.py` measures that claim
properly: it renders each window as text and asks one LLM for a verdict, over
the identical 45 scenarios.

**It has not been run, and no numbers are claimed for it**, because no API key
was available in the environment where these results were produced. The script
deliberately **refuses to run on `MockProvider`** rather than emit figures that
look like an LLM baseline and are not one. With a key:

```bash
ANTHROPIC_API_KEY=... python llm_baseline.py --provider anthropic   # < $1
```

---

## 7. Reproducibility

- **Everything is seeded.** `--seed 42` regenerates the dataset bit-identically
  (`test_same_seed_is_bit_identical`). The research run reproduces
  epoch-for-epoch; this was verified by re-running it from a cleared registry
  and getting identical validation curves.
- **No credentials, no private data.** The dataset is synthetic and generated
  locally. `artifacts/data/manifest.json` records the config and a SHA-256 of
  every file.
- **No API key needed.** `get_provider()` falls back to `MockProvider`, which
  returns each agent's deterministic findings unchanged. With an LLM
  configured the prose improves; **the measured metrics do not move**, because
  no metric passes through the LLM.
- **127 tests**, including causality, leakage, determinism and the
  fabricated-evidence catch: `python -m pytest`.

Leakage controls are concentrated in `data/preprocessing.py` and enforced by
tests: chronological splits with a one-horizon purge gap at each boundary,
train-only scaler and imputation constants, forward-fill only, windows that
never span downtime. `test_features_do_not_depend_on_the_future` recomputes
every feature from a truncated history and asserts the values are unchanged —
the test that would catch a `center=True` rolling window or a backfill.

See [`REPRODUCTION.md`](REPRODUCTION.md) for exact commands, versions, runtimes
and expected output.

---

## 8. Main failure mode, and the hot take

### The failure mode that cost the most

**Two agents describing the same physical thing in different vocabularies.**
The investigator reported `temp_excess` (temperature above ambient — the
better signal, because it removes the weather confound). The failure
signatures were written in plain sensor names: `temperature`. The verifier
compared the two, found no overlap, and marked a **correct** bearing diagnosis
as unsupported — a confident, well-evidenced, entirely right prediction,
rejected on a naming mismatch.

Nothing threw. Every test passed. The system quietly refused its own best
work, and it took reading a rendered report to notice. The fix is a canonical
channel vocabulary shared by both agents (`diagnosis.SIGNATURE_ALIASES`), which
lifted the signature match on that case from 1/3 to 2/3.

Two more of the same shape turned up, both found by reading output rather than
by any test:

| Defect | Symptom | Cause | Fix |
|---|---|---|---|
| Reliability curve | **13% confidence on a 99.6% prediction** | quantile bins on a 1.4%-positive set: the top bin spanned the top 8% of all rows and drowned in ordinary ones | measure *tail* precision — "of validation cases scoring at least this high, how many failed?" Same case now reads 86% |
| Verification scope | **25 of 45 evaluation cases marked FAIL** | diagnosis checks ran on machines the model was *not* alerting about, failing for want of a diagnosis that was never asserted | scope checks to the claim being made; `n/a` when there is no alert. Verdicts now read **PASS 28 / PASS+WARN 15 / FAIL 2**, and both remaining FAILs are real catches |
| Two fleet views | **the assistant named a stopped pump as the plant's highest risk (69.5%) while the dashboard showed the same pump as `down`** | both were "right": the dashboard reads the canonical window set, which never scores a window spanning downtime; `fleet_scores` — behind the assistant and the incident picker — checked only whether the *last* step was downtime, so it scored a sequence model on a lookback straddling an outage | apply the rule the deployed model was *trained* under, which differs by kind: whole lookback clean for a sequence model, current row usable for a tabular one. The two views now return identical scores for identical machines (`test_fleet_scores_and_overview_agree`) |

### Hot take

**Adding an agent is a hypothesis, and most teams never test theirs.**

The hypothesis contest is the best-engineered thing in this repo. Two advocates
argue from a shared, recomputable evidence base; neither can invent a fact;
each must state what would change its mind; an adjudicator decides on
arithmetic. It reads like exactly the multi-agent reasoning an architecture
diagram promises.

It is also worth **+0.0000 F1**, and the first version of it was worth
**−0.039** — it overturned four correct predictions, including a real failure
one hour from happening. I only know that because I built the ablation
(`ablate_adjudication.py`) instead of assuming.

For most of the suite the reason is that the idea addresses a failure mode the
system does not have: the model rejects 21 of 22 confounders on its own, so
there is nothing to overturn. But the twenty-second is worse than inert. On
S37 — a pressure-transducer spike scored at 99.1% — the confound advocate,
whose whole job is spikes, scored the benign explanation **0.00**, and ten of
ten verification checks passed a $350 physical intervention on a healthy pump.

The component did not merely fail to help. It supplied the *appearance* of
scrutiny — a contest, a margin, a ten-point audit — over a decision nothing had
actually challenged. That is a worse failure than not having built it, because
an inert component is ignorable and a confidently empty one is trusted.

Three lessons I would carry to the next build:

1. **Every agent needs an ablation, not a rationale.** "It makes the reasoning
   explicit" is a story. "+0.0000 F1, 0 verdicts changed" is a finding. Only
   one of them should decide whether the component ships.
2. **Measure where the headroom actually is before designing for it.** The
   headroom here was recall (47.8%), and the biggest single win in the whole
   project — **+0.197 PR-AUC** — came from causal rolling features, not from
   any agent. Every architectural change in the bake-off put together is worth
   less than a sixth of that.
3. **Keep the component if it changes the decision, not because it changes the
   score.** The contest stays because a `contested` verdict routes to *inspect*
   rather than *repair*, and because a planner seeing "0.92 vs 0.00" trusts it
   differently from "0.56 vs 0.48". Those are honest reasons. "It improved
   accuracy" would not have been.

The same discipline caught the other defects in this project, all silent, none
surfaced by a test: a vocabulary mismatch that rejected correct diagnoses, a
reliability curve that reported 13% confidence on a 99.6% prediction, a
verifier that failed 25 of 45 cases for the crime of being healthy, a ten-point
audit that passed a physical intervention on a healthy pump because it checks
whether a counter-argument was *made*, not whether it was *made well*, and two
fleet views that disagreed about which machine was the plant's biggest risk
because each applied a defensible — and different — definition of "scorable".

In a system where agents pass structured state to each other, the realistic
failure is never a model that lies. It is two components that are each correct
and do not agree; a metric that is arithmetically right and quietly measuring
the wrong thing; and a check that is satisfied by the shape of an argument
rather than its content.

---

## 9. Layout

```
predictops/
├── data/           generator, schemas, splits, leakage controls
├── ml/             features, models (LSTM/TFT/trees), training, metrics,
│                   bundle, and service.py -- the agent/ML boundary
├── agents/         the 12 agents + orchestrator (PredictOpsEngine)
│                   assistant.py  grounded Q&A, refuses to authorise work
│                   workflow.py   the agent graph as data: contracts,
│                                 validation and execution
│                   evidence.py   shared, recomputable fact toolkit
│                   context.py    the machine dossier
│                   hypothesis.py the two advocates + adjudicator
├── simulation/     intervention catalogue + counterfactual environment
├── experiments/    SQLite registry, runner, changelog generator
├── evaluation/     canonical eval harness, scenario suite, scenario runner
├── llm/            provider abstraction (Anthropic / OpenAI / Mock)
├── api/            FastAPI + WebSocket agent-activity stream
└── reporting.py    the incident report a planner actually reads
frontend/           React + TypeScript dashboard (9 pages)
tests/              127 tests
```

**Written for this hackathon.** Pre-existing components used as libraries:
PyTorch, scikit-learn, XGBoost/LightGBM, pandas, FastAPI, React, Recharts.
The TFT is implemented from the architecture in this repo (not a wrapper
around a forecasting library) so the model can expose its variable-selection
weights and attention to the investigation agent.

---

## 10. Safety

Decision support only. Nothing in this repo actuates machinery, and there is
no interface through which it could: interventions are catalogue entries whose
effects are applied to a **synthetic copy** of the telemetry.

- Consequential actions carry `requires_approval` and an explicit approval
  gate, checked by C9 and by tests.
- Simulated outcomes are labelled `is_simulated` in the data model, `[simulated]`
  in the report, and carry a standing caveat that they rank actions rather
  than forecast them.
- A `FAIL` verdict blocks the "safe to act" flag regardless of how confident
  the model is.
