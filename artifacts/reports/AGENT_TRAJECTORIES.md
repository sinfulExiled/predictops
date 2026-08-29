# Agent Trajectories

One section per agent: the instructions it runs under, a real execution, the tools that answered it, and the feedback that shaped what happened next. Records come straight from `artifacts/experiments/experiments.db`.

Every agent follows the same contract — `investigate()` computes findings with deterministic tools, then an optional `narrate()` pass phrases them. No number in any output below passed through a language model.

---

## data_scientist

**Brief.** Profile the telemetry, find leakage and imbalance risks, and recommend the
feature set and training window.

**Instructions given to the language model** (narrative pass only):

> You are a senior data scientist reviewing an industrial telemetry dataset
> before any model is trained. You are given a computed profile. Summarise the
> three findings that most affect modelling choices. Do not invent statistics;
> use only the numbers provided.

**Tools available.** `pandas.profile`, `corr.point_biserial`, `split.audit`, `leakage.scan`

**Run** `main` · step 1 · 1.21s · retries 0

**Input.** `(none)`

**Action.** Profiled the dataset and derived the modelling plan.

**Reason.** 345,600 rows across 80 machines at 10 min resolution; 1.13% positive.

**Result.**

```json
{
  "leakage_audit": {
    "leaky_columns_in_feature_list": [],
    "splits_chronological": true,
    "split_bounds": {
      "purge": {
        "start": "2025-03-18 18:00:00",
        "end": "2025-03-23 11:50:00",
        "rows": 5760,
        "positives": 41
      },
      "test": {
        "start": "2025-03-23 12:00:00",
        "end": "2025-03-30 23:50:00",
        "rows": 86400,
        "positives": 1021
      },
      "train": {
        "start": "2025-03-01 00:00:00",
        "end": "2025-03-18 17:50:00",
        "rows": 204480,
        "positives": 2304
      },
      "val": {
        "start": "2025-03-19 00:00:00",
        "end": "2025-03-23 05:50:00",
        "rows": 48960,
        "positives": 666
      }
    },
    "verdict": "clean"
  },
  "class_balance": {
    "train_positive_rate": 0.01127,
    "imbalance_ratio": 87.8,
    "train_positives": 2304
  },
  "recommended_plan": {
    "selection_metric": "pr_auc",
    "primary_report_metric": "f1",
    "feature_sets_to_ablate": [
      "raw",
      "engineered"
    ],
    "try_sequence_models": true,
    "lookback_steps": 36,
    "balance_training_data": true
  }
}
```

---

## model_researcher

**Brief.** Plan and run model experiments, then select the best candidate on measured
validation PR-AUC.

**Instructions given to the language model** (narrative pass only):

> You are an ML researcher writing up a model bake-off. You are given the
> measured results of every experiment that ran. Explain which model was
> selected and why, referring only to the numbers provided. Never claim a model
> is better than its measured score shows.

**Tools available.** `experiment_runner.train`, `experiment_runner.evaluate`, `registry.record`, `registry.compare`

**Run** `main` · step 2 · 1056.55s · retries 0

**Input.** `runner=<ExperimentRunner>, quick=False`

**Action.** Ran 7 experiments and selected xgboost on engineered features.

**Reason.** xgboost on engineered features selected on validation PR-AUC 0.6112; ensemble
lstm_engineered + tft_engineered scored 0.6167 but the +0.0055 gain is inside
the 0.01 tolerance, so the simpler model wins the tie

**Self-reported check.** Selection used validation PR-AUC only; test metrics were not consulted for
model choice.

**Result.**

```json
{
  "selection": {
    "experiment_id": 39,
    "name": "xgboost on engineered features",
    "model": "xgboost",
    "feature_set": "engineered",
    "val_pr_auc": 0.611207,
    "test_f1": 0.536369,
    "rationale": "xgboost on engineered features selected on validation PR-AUC 0.6112; ensemble lstm_engineered + tft_engineered scored 0.6167 but the +0.0055 gain is inside the 0.01 tolerance, so the simpler model wins the tie",
    "selected_on": "validation pr_auc"
  },
  "candidates_run": [
    "baseline",
    "xgboost_raw",
    "xgboost_engineered",
    "lstm_raw",
    "lstm_engineered",
    "tft_engineered",
    "ensemble"
  ]
}
```

---

## predictor

**Brief.** Ask the model service what is likely to happen to one machine at one
timestamp, and report it unchanged.

**Tools available.** `model_service.predict`, `model_service.raw_window`

**Run** `incident-20260829-190728` · step 1 · 0.06s · retries 0

**Input.** `service=<ModelService>, machine_id=COMPRESSOR-066, timestamp=<Timestamp>`

**Action.** Scored COMPRESSOR-066 at 2025-03-29 01:30:00.

**Reason.** failure probability 0.996 against an alert threshold of 0.425 (investigate
from 0.191) -> high

**Self-reported check.** Probability, type, ETA and confidence are all fitted-model outputs.

**Result.**

```json
{
  "failure_probability": 0.9959,
  "alert": true,
  "threshold": 0.425,
  "prediction_window_hours": {
    "eta_hours": 2.97,
    "window_low_h": 1.36,
    "window_high_h": 4.58
  },
  "failure_type": "bearing_degradation",
  "confidence": 0.8615,
  "confidence_basis": "measured share of validation cases scoring at least this high that were genuinely followed by a failure"
}
```

---

## context

**Brief.** Assemble the machine dossier: what is normal for this unit, what has been done
to it, and what it has failed with before.

**Tools available.** `machines.lookup`, `maintenance.history`, `failures.history`, `telemetry.baseline_stats`

**Run** `incident-20260829-190728` · step 2 · 0.05s · retries 0

**Input.** `machine_id=COMPRESSOR-066, timestamp=<Timestamp>`

**Action.** Assembled the dossier for COMPRESSOR-066.

**Reason.** COMPRESSOR, 158 h since service, 2 prior failure(s)

**Result.**

```json
{
  "machine_type": "COMPRESSOR",
  "in_run_in_period": false,
  "hours_since_service": 158.0,
  "prior_failure_types": {
    "motor_overheating": 1,
    "pressure_loss": 1
  },
  "recurring_mode": "motor_overheating",
  "notes": [
    "has failed with motor overheating 1x before -- a repeat is more likely than the fleet base rate"
  ]
}
```

---

## investigator

**Brief.** Establish the factual record: what moved, by how much, in what operating
context, and what it resembles.

**Instructions given to the language model** (narrative pass only):

> You are a reliability engineer summarising what a machine's telemetry did over
> the last few hours, for colleagues who will argue about what it means. State
> only what the evidence says. Do not diagnose, and do not state any number that
> is not in the evidence.

**Tools available.** `telemetry.window`, `evidence.channel_movements`, `model.attribution`, `signature_library.nearest`

**Run** `incident-20260829-190728` · step 3 · 0.05s · retries 0

**Input.** `service=<ModelService>, machine_id=COMPRESSOR-066, timestamp=<Timestamp>, library=<SignatureLibrary>`

**Action.** Recorded the factual state of COMPRESSOR-066.

**Reason.** 5 channel movement(s) recorded: vibration, temperature, temp_excess, load,
rpm_instability_1h

**Self-reported check.** 5 evidence item(s), each recomputable.

**Result.**

```json
{
  "evidence": [
    {
      "id": "E1",
      "claim": "Vibration rose 43% over the last 6 hours",
      "channel": "vibration",
      "metric": "pct_change",
      "value": 42.9091,
      "unit": "%",
      "direction": "up",
      "source": "telemetry[COMPRESSOR-066, 2025-03-28 19:40:00 .. 2025-03-29 01:30:00]",
      "recompute": {
        "fn": "pct_change",
        "channel": "vibration",
        "hours": 6.0
      }
    },
    {
      "id": "E2",
      "claim": "Temperature rose 5.9 deg C over the last 6 hours",
      "channel": "temperature",
      "metric": "abs_change",
      "value": 5.8992,
      "unit": "deg C",
      "direction": "up",
      "source": "telemetry[COMPRESSOR-066, 2025-03-28 19:40:00 .. 2025-03-29 01:30:00]",
      "recompute": {
        "fn": "abs_change",
        "channel": "temperature",
        "hours": 6.0
      }
    },
    {
      "id": "E3",
      "claim": "Temp excess rose 7.7 deg C above ambient over the last 6 hours",
      "channel": "temp_excess",
      "metric": "abs_change",
      "value": 7.6998,
      "unit": "deg C above ambient",
      "direction": "up",
      "source": "telemetry[COMPRESSOR-066, 2025-03-28 19:40:00 .. 2025-03-29 01:30:00]",
      "recompute": {
        "fn": "abs_change",
        "channel": "temp_excess",
        "hours": 6.0
      }
    },
    {
      "id": "E4",
      "claim": "Load fell 0.1 fraction over the last 6 hours",
      "channel": "load",
      "metric": "abs_change",
      "value": -0.0565,
      "unit": "fraction",
      "direction": "down",
      "source": "telemetry[COMPRESSOR-066, 2025-03-28 19:40:00 .. 2025-03-29 01:30:00]",
      "recompute": {
        "fn": "abs_change",
        "channel": "load",
        "hours": 6.0
      }
    },
    {
      "id": "E5",
      "claim": "Rpm_instability_1h rose 81% over the last 6 hours",
      "channel": "rpm_instability_1h",
      "metric": "pct_change",
      "value": 81.1602,
      "unit": "%",
      "direction": "up",
      "source": "telemetry[COMPRESSOR-066, 2025-03-28 19:40:00 .. 2025-03-29 01:30:00]",
      "recompute": {
        "fn": "pct_change",
        "channel": "rpm_instability_1h",
        "hours": 6.0
      }
    }
  ],
  "operating_context": {
    "load_change_pct_3h": 2.03,
    "ambient_change_c_3h": -3.5,
    "load_stable": true,
    "ambient_rising": false
  },
  "summary": "5 channel movement(s) recorded: vibration, temperature, temp_excess, load, rpm_instability_1h"
}
```

---

## degradation_advocate

**Brief.** Argue that the machine is developing a fault, name the mode, and state what
would refute it.

**Instructions given to the language model** (narrative pass only):

> You are a reliability engineer arguing that a machine is developing a fault.
> You are given computed evidence and scores. Make the strongest honest case,
> using only those numbers, and state plainly what evidence would change your
> mind. Do not invent measurements.

**Tools available.** `evidence.channel_movements`, `evidence.monotonicity`, `signature.match`, `history.nearest_failures`, `model.attribution`

**Run** `incident-20260829-190728` · step 4 · 0.00s · retries 0

**Input.** `builder=<EvidenceBuilder>, prediction=<dict>, context=<dict>, neighbours=<list>`

**Action.** Argued degradation at 0.92.

**Reason.** bearing degradation is developing

**Result.**

```json
{
  "score": 0.9223,
  "conclusion": "bearing degradation is developing",
  "factors": {
    "model_probability": 0.9959,
    "best_mode_score": 0.6966,
    "trend_persistence": 1.0,
    "load_is_flat": true,
    "machine_has_failed_this_way_before": false
  },
  "would_change_my_mind": "A load or ambient increase of the same shape and timing as the channel movements, or a single-sample spike rather than a sustained trend, would undercut this.",
  "ranked_types": [
    {
      "failure_type": "bearing_degradation",
      "score": 0.6966,
      "classifier_probability": 0.9932,
      "signature_match": 0.667,
      "matched_channels": [
        "vibration",
        "temperature"
      ],
      "expected_signature": {
        "vibration": "up",
        "temperature": "up",
        "current": "up"
      },
      "historical_vote": 0.0,
      "evidence_ids": [
        "E1",
        "E2",
        "E3",
        "E6"
      ]
    },
    {
      "failure_type": "motor_overheating",
      "score": 0.3456,
      "classifier_probability": 0.0033,
      "signature_match": 0.75,
      "matched_channels": [
        "vibration",
        "temperature",
        "rpm_instability"
      ],
      "expected_signature": {
        "vibration": "up",
        "temperature": "up",
        "current": "up",
        "rpm_instability": "up"
      },
      "historical_vote": 0.5945,
      "evidence_ids": [
        "E1",
        "E2",
        "E3",
        "E5",
        "E6"
      ]
    },
    {
      "failure_type": "electrical_fault",
      "score": 0.219,
      "classifier_probability": 0.0,
      "signature_match": 0.6,
      "matched_channels": [
        "vibration",
        "temperature",
        "rpm_instability"
      ],
      "expected_signature": {
        "vibration": "up",
        "temperature": "up",
        "current": "up",
        "voltage": "down",
        "rpm_instability": "up"
      },
      "historical_vote": 0.195,
      "evidence_ids": [
        "E1",
        "E2",
        "E3",
        "E5",
        "E6"
      ]
    },
    {
      "failure_type": "pressure_loss",
      "score": 0.1929,
      "classifier_probability": 0.0017,
      "signature_match": 0.5,
      "matched_channels": [
        "vibration",
        "temperature"
      ],
      "expected_signature": {
        "vibration": "up",
        "temperature": "up",
        "current": "down",
        "pressure": "down"
      },
      "historical_vote": 0.2104,
      "evidence_ids": [
        "E1",
        "E2",
        "E3",
        "E6"
      ]
    },
    {
      "failure_type": "pump_cavitation",
      "score":
  ... (truncated; full record in the registry)
```

**Constraint.** This agent extends the shared evidence record built by the investigator; it cannot introduce a measurement of its own, and the verifier re-derives every item it cites from raw telemetry.

---

## confound_advocate

**Brief.** Argue that the reading has a benign explanation -- production load, ambient
heat, a sensor glitch, or post-service run-in.

**Instructions given to the language model** (narrative pass only):

> You are a sceptical operations engineer. Your job is to find the innocent
> explanation for an alarm before a crew is sent out. You are given computed
> evidence. Make the strongest honest case that nothing is wrong, using only
> those numbers, and state what would change your mind. Do not invent
> measurements.

**Tools available.** `evidence.channel_movements`, `evidence.peak_ratio`, `evidence.monotonicity`, `context.dossier`, `telemetry.load_profile`

**Run** `incident-20260829-190728` · step 5 · 0.00s · retries 0

**Input.** `builder=<EvidenceBuilder>, context=<dict>, prediction=<dict>`

**Action.** Argued a benign explanation at 0.00.

**Reason.** no benign explanation is available -- nothing in the operating context
accounts for the movement

**Result.**

```json
{
  "score": 0.0,
  "conclusion": "no benign explanation is available -- nothing in the operating context accounts for the movement",
  "alternative_explanations": [],
  "would_change_my_mind": "A sustained multi-channel trend continuing after the load and ambient returned to normal would defeat this."
}
```

**Constraint.** This agent extends the shared evidence record built by the investigator; it cannot introduce a measurement of its own, and the verifier re-derives every item it cites from raw telemetry.

---

## adjudicator

**Brief.** Weigh the degradation case against the benign case and decide whether this is
an alert, on the numbers.

**Tools available.** `cases.compare`, `thresholds.apply`

**Run** `incident-20260829-190728` · step 6 · 0.00s · retries 0

**Input.** `degradation=<dict>, confound=<dict>, prediction=<dict>`

**Action.** Adjudicated: alert.

**Reason.** the degradation case survives by 0.92 (degradation 0.92 vs benign 0.00)

**Self-reported check.** decided on computed scores; margin +0.922

**Result.**

```json
{
  "decision": "alert",
  "degradation_score": 0.9223,
  "confound_score": 0.0,
  "margin": 0.9223,
  "rationale": "the degradation case survives by 0.92 (degradation 0.92 vs benign 0.00)",
  "changed_the_model_verdict": false,
  "recommend_physical_work": true
}
```

**Human checkpoint.** An `overturned` or `insufficient_evidence` decision stops the workflow proposing any physical work; a `contested` decision downgrades the plan to inspection only. The remediation agent is gated on this value.

---

## remediation

**Brief.** Propose an ordered plan of approved interventions, matched to the diagnosis
and its confidence.

**Instructions given to the language model** (narrative pass only):

> You are a maintenance planner. You are given a diagnosis and a list of
> already-selected approved actions in order. Summarise the plan for a
> supervisor in plain language. Do not add, remove or reorder actions, and do
> not invent any action that is not in the list.

**Tools available.** `catalogue.applicable`, `catalogue.get`, `preconditions.evaluate`

**Run** `incident-20260829-190728` · step 7 · 0.00s · retries 0

**Input.** `prediction=<dict>, investigation=<dict>, adjudication=<dict>, context=<dict>`

**Action.** Proposed 4 approved action(s), mode=act.

**Reason.** diagnosis bearing_degradation, probability 1.00; 3 action(s) need human
approval

**Self-reported check.** Every action id was checked against the approved catalogue.

**Result.**

```json
{
  "diagnosis": "bearing_degradation",
  "mode": "act",
  "plan": [
    {
      "order": 1,
      "intervention_id": "reduce_load_70",
      "title": "Reduce load to 70%",
      "detail": "Throttle the duty setpoint to 70% of current. Buys time on a developing fault without stopping production.",
      "why": "reversible and reduces the driver of bearing degradation while the inspection is arranged",
      "risk": "low",
      "cost_usd": 180.0,
      "downtime_hours": 0.0,
      "requires_approval": true,
      "is_diagnostic": false,
      "preconditions": [
        "production schedule allows reduced throughput"
      ],
      "expected_effect": {
        "load": {
          "mul": 0.7
        }
      }
    },
    {
      "order": 2,
      "intervention_id": "inspect_bearing",
      "title": "Inspect bearing within 4 hours",
      "detail": "Hands-on inspection: temperature gun, listening stick, grease condition. Diagnostic: confirms before committing to a replacement.",
      "why": "confirms the diagnosis before any irreversible work; within 1 h",
      "risk": "low",
      "cost_usd": 150.0,
      "downtime_hours": 0.0,
      "requires_approval": false,
      "is_diagnostic": true,
      "preconditions": [],
      "expected_effect": "no telemetry change (diagnostic)"
    },
    {
      "order": 3,
      "intervention_id": "replace_bearing",
      "title": "Replace bearing",
      "detail": "Planned shutdown and bearing change. The definitive fix once inspection confirms degradation.",
      "why": "the definitive fix, to be carried out after inspection confirms the fault",
      "risk": "high",
      "cost_usd": 2600.0,
      "downtime_hours": 4.0,
      "requires_approval": true,
      "is_diagnostic": false,
      "preconditions": [
        "inspection has confirmed bearing degradation",
        "replacement bearing in stock"
      ],
      "expected_effect": {
        "vibration": {
          "mul": 0.35
        },
        "temperature": {
          "add": -12.0
        },
        "current": {
          "mul": 0.88
        }
      }
    },
    {
      "order": 4,
      "intervention_id": "controlled_shutdown",
      "title": "Controlled shutdown",
      "detail": "Stop the machine in a controlled way. Reserved for imminent failure where continued running risks secondary damage.",
      "why": "probability 1.00 with an estimated 3.0 h to failure leaves little margin; a controlled stop avoids secondary damage",
      "risk": "high",
      "cost_usd": 5200.0,
      "downtime_hours": 6.0,
      "requires_approval": true,
      "is_diagnostic": fals
  ... (truncated; full record in the registry)
```

**Human checkpoint.** PredictOps proposes; it does not act. No action is executed without a named human approver. Awaiting approval for: reduce_load_70, replace_bearing, controlled_shutdown.

---

## simulator

**Brief.** Roll the machine forward under each proposed action and a do-nothing control,
and score every counterfactual.

**Tools available.** `machine_environment.rollout`, `features.recompute`, `bundle.score`

**Run** `incident-20260829-190728` · step 8 · 0.53s · retries 0

**Input.** `bundle=<ModelBundle>, machine_id=COMPRESSOR-066, timestamp=<Timestamp>, plan=<list>, baseline_probability=0.9959`

**Action.** Simulated 3 action(s) plus a control arm over 3 h.

**Reason.** no action -> 0.238; best action (controlled_shutdown) -> 0.000

**Self-reported check.** All figures are model scores on synthetic telemetry and are labelled
simulated; the control arm uses the identical rollout so the delta isolates
the intervention.

**Result.**

```json
{
  "no_action": {
    "failure_probability_simulated": 0.2384,
    "channels": {
      "temperature": 59.839,
      "vibration": 8.361,
      "current": 26.364,
      "pressure": 6.513,
      "load": 0.609
    },
    "is_simulated": true
  },
  "arms": [
    {
      "intervention_id": "reduce_load_70",
      "title": "Reduce load to 70%",
      "simulated": true,
      "is_simulated": true,
      "failure_probability_simulated": 0.0868,
      "delta_vs_no_action": -0.1516,
      "delta_vs_now": -0.9091,
      "relative_reduction_pct": 63.6,
      "projected_channels": {
        "temperature": 52.234,
        "vibration": 7.383,
        "current": 21.039,
        "pressure": 5.862,
        "load": 0.426
      },
      "cost_usd": 180.0,
      "downtime_hours": 0.0,
      "risk_reduction_per_1k_usd": 0.8422
    },
    {
      "intervention_id": "inspect_bearing",
      "title": "Inspect bearing within 4 hours",
      "is_simulated": true,
      "simulated": false,
      "reason_not_simulated": "diagnostic action -- gathers information, changes no telemetry, so there is nothing to simulate",
      "failure_probability_simulated": null,
      "delta_vs_no_action": null
    },
    {
      "intervention_id": "replace_bearing",
      "title": "Replace bearing",
      "simulated": true,
      "is_simulated": true,
      "failure_probability_simulated": 0.0009,
      "delta_vs_no_action": -0.2375,
      "delta_vs_now": -0.995,
      "relative_reduction_pct": 99.6,
      "projected_channels": {
        "temperature": 47.839,
        "vibration": 2.926,
        "current": 23.201,
        "pressure": 6.513,
        "load": 0.609
      },
      "cost_usd": 2600.0,
      "downtime_hours": 4.0,
      "risk_reduction_per_1k_usd": 0.0913
    },
    {
      "intervention_id": "controlled_shutdown",
      "title": "Controlled shutdown",
      "simulated": true,
      "is_simulated": true,
      "failure_probability_simulated": 0.0,
      "delta_vs_no_action": -0.2384,
      "delta_vs_now": -0.9959,
      "relative_reduction_pct": 100.0,
      "projected_channels": {
        "temperature": 34.488,
        "vibration": 5.102,
        "current": 7.382,
        "pressure": 3.582,
        "load": 0.0
      },
      "cost_usd": 5200.0,
      "downtime_hours": 6.0,
      "risk_reduction_per_1k_usd": 0.0458
    }
  ],
  "best_by_risk": "controlled_shutdown",
  "best_by_value": "reduce_load_70",
  "simulation_shows_improvement": true
}
```

---

## verifier

**Brief.** Re-derive every claim from raw telemetry and challenge the diagnosis, the plan
and the simulation.

**Tools available.** `telemetry.window`, `evidence.recompute`, `signature.compare`, `catalogue.validate`, `narrative.scan_numbers`

**Run** `incident-20260829-190728` · step 9 · 0.02s · retries 0

**Input.** `bundle=<ModelBundle>, prediction=<dict>, investigation=<dict>, remediation=<dict>, simulation=<dict>, adjudication=<dict>`

**Action.** Ran 10 verification checks -> PASS.

**Reason.** all checks passed

**Self-reported check.** PASS

**Result.**

```json
{
  "verdict": "PASS",
  "headline": "all checks passed",
  "checks": [
    {
      "id": "C1",
      "check": "evidence recomputes from raw telemetry",
      "status": "pass",
      "detail": "7 item(s) re-derived independently; 0 mismatch(es)",
      "mismatches": [],
      "n_evidence": 7
    },
    {
      "id": "C2",
      "check": "evidence is consistent with the diagnosis",
      "status": "pass",
      "detail": "2/3 signature channels for bearing_degradation are present",
      "expected": {
        "vibration": "up",
        "temperature": "up",
        "current": "up"
      },
      "observed": {
        "vibration": "up",
        "temperature": "up",
        "load": "down",
        "rpm_instability": "up"
      },
      "match_ratio": 0.667
    },
    {
      "id": "C3",
      "check": "prediction clears the decision threshold",
      "status": "pass",
      "detail": "probability 0.9959 vs validation-tuned threshold 0.4250",
      "probability": 0.9959,
      "threshold": 0.425
    },
    {
      "id": "C4",
      "check": "confidence is empirically grounded",
      "status": "pass",
      "detail": "confidence 0.862 read from the validation reliability curve",
      "confidence": 0.8615,
      "basis": "measured share of validation cases scoring at least this high that were genuinely followed by a failure"
    },
    {
      "id": "C5",
      "check": "benign alternatives considered",
      "status": "pass",
      "detail": "2 alternative(s) evaluated; 0 still plausible",
      "alternatives": [
        {
          "explanation": "production load change",
          "evidence": "load moved only +2.0% over 3 h",
          "verdict": "rejected -- load is flat"
        },
        {
          "explanation": "high ambient temperature",
          "evidence": "ambient moved -3.5 C over 3 h",
          "verdict": "rejected -- ambient is stable"
        }
      ]
    },
    {
      "id": "C6",
      "check": "all proposed actions are in the approved catalogue",
      "status": "pass",
      "detail": "4 action(s) checked",
      "illegal": []
    },
    {
      "id": "C7",
      "check": "simulation beats its own do-nothing control",
      "status": "pass",
      "detail": "control 0.238 vs best action 0.000 (-0.238)",
      "control": 0.2384,
      "best": "controlled_shutdown"
    },
    {
      "id": "C8",
      "check": "narrative quotes only supported numbers",
      "status": "pass",
      "detail": "11 number(s) in the narrative; 0 not traceable to evidence",
      "unsupported": []
    },
    {
      "id": "C9",
      "check": "consequential 
  ... (truncated; full record in the registry)
```

**Human checkpoint.** A `FAIL` verdict clears the `safe_to_act` flag, and any action marked `requires_approval` stays behind the approval gate regardless of the model's confidence.

---

## assistant

**Brief.** Answer operator questions strictly from computed artifacts, citing every
number, and refuse anything it cannot ground.

**Instructions given to the language model** (narrative pass only):

> You are the assistant inside a predictive-maintenance tool. You are given a
> set of FACTS retrieved from the system's own records, and a draft answer.
> Rewrite the draft so it reads naturally for a maintenance planner. You may not
> introduce any number, machine name or claim that is not in the FACTS. If the
> draft says the system does not know something, keep that.

**Tools available.** `assistant.route`, `fleet.scores`, `registry.experiments`, `reports.evaluation`, `reports.ablation`, `catalogue.list`, `engine.run_incident`

**Run** `assistant-demo` · step 3 · 0.01s · retries 0

**Input.** `question=approve the repair, engine=<PredictOpsEngine>`

**Action.** Refused an action request.

**Reason.** outside the assistant's authority

**Result.**

```json
{
  "intent": "refused_action",
  "answer": "I can't approve, schedule or carry out physical work. This system proposes and simulates; a named human approves in the Remediation Simulator, and nothing here actuates a machine. I can show you the proposed plan and what the simulation says it would buy.",
  "citations": [],
  "grounded": true,
  "refused": true,
  "action": null
}
```

**Constraint.** The assistant cannot originate a fact. It routes the question deterministically, retrieves from computed artifacts with a citation per number, and discards any LLM rephrasing that introduces a number not in those facts. Requests to approve, schedule or carry out physical work are refused before retrieval runs.

_3 executions of this agent are recorded across the exported runs._
