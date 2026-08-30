## Improvement Changelog

Run `main`. Every figure below was read back from the experiment registry (`artifacts/experiments/experiments.db`); none is hand-entered.

Models are **selected on validation PR-AUC**. Test F1 is reported for the same frozen threshold and is never used to choose between candidates.

| Stage | What was tried and why | Val PR-AUC | Test F1 | vs baseline | Decision / learning |
|---|---|---|---|---|---|
| **Baseline** | threshold rule on vibration / temperature / current<br/><sub>A per-machine z-score alarm is what the plant already runs; establish where that leaves us.</sub> | 0.1446 | 0.2570 | -- | `reference` Establishes the operating point the plant already has. |
| **Iteration 1** | xgboost on raw features<br/><sub>Do gradient-boosted trees over raw features beat the alarm rule without any sequence modelling?</sub> | 0.3099 | 0.4430 | +0.1860 | `kept` PR-AUC 0.3099 vs baseline 0.1446 (+0.1653). Trees on raw channels already beat the alarm rule, so most of the baseline's weakness was the fixed threshold, not the sensors. |
| **Iteration 2** | xgboost on engineered features<br/><sub>Do gradient-boosted trees over engineered features beat the alarm rule without any sequence modelling?</sub> | 0.5072 | 0.5118 | +0.2548 | `kept` PR-AUC 0.5072 vs raw features 0.3099 (+0.1973). Causal rolling statistics and load/ambient normalisation are worth more than any change of architecture tried below. |
| **Iteration 3** | LSTM on raw channels<br/><sub>Does an LSTM over a 36-step window on raw channels capture degradation the tabular view misses?</sub> | 0.3640 | 0.4313 | +0.1743 | `removed` PR-AUC 0.3640 vs best tabular 0.5072 (-0.1432). Learning the temporal structure from raw sequence does not beat handing a tree the same structure explicitly. |
| **Iteration 4** | LSTM on engineered channels<br/><sub>Does an LSTM over a 36-step window on engineered channels capture degradation the tabular view misses?</sub> | 0.5542 | 0.4937 | +0.2367 | `kept` PR-AUC 0.5542 vs best tabular 0.5072 (+0.0470). The sequence view finds structure the tabular one misses. |
| **Iteration 5** | TFT on engineered channels<br/><sub>Does variable selection plus attention beat a plain LSTM on the same channels?</sub> | 0.5583 | 0.4659 | +0.2089 | `removed` PR-AUC 0.5583 vs LSTM 0.5542. No material gain over the LSTM at this data scale -- the extra capacity is not paying for itself. |
| **Iteration 6** | ensemble lstm_engineered + tft_engineered<br/><sub>Do the two sequence models make different mistakes? If so a validation-weighted blend should beat either alone.</sub> | 0.5855 | 0.5142 | +0.2572 | `kept` PR-AUC 0.5855 vs best member 0.5583. The blend adds real signal. |

### Operational view

Row-level F1 is the model-selection metric. What a maintenance planner actually feels is the next table: how many real failures were caught, how much warning they got, and how many times the crew was sent out for nothing.

| Stage | Model | Events caught | Mean early warning (h) | False alarms / machine / day | Precision | Recall |
|---|---|---|---|---|---|---|
| Baseline | threshold_baseline | 19/26 (73%) | 4.39 | 2.059 | 0.211 | 0.329 |
| Iteration 1 | xgboost | 20/26 (77%) | 4.52 | 0.876 | 0.453 | 0.433 |
| Iteration 2 | xgboost | 21/26 (81%) | 4.51 | 0.687 | 0.542 | 0.485 |
| Iteration 3 | lstm | 18/26 (69%) | 4.46 | 1.498 | 0.368 | 0.521 |
| Iteration 4 | lstm | 17/26 (65%) | 4.45 | 0.636 | 0.544 | 0.452 |
| Iteration 5 | tft | 15/26 (58%) | 4.48 | 0.445 | 0.591 | 0.384 |
| Iteration 6 | ensemble | 16/26 (62%) | 4.72 | 0.460 | 0.616 | 0.441 |

### Where the improvement came from

- `xgboost on raw features` -> `xgboost on engineered features`: +0.1973 val PR-AUC -- **gain**
- `LSTM on raw channels` -> `LSTM on engineered channels`: +0.1902 val PR-AUC -- **gain**
- `threshold rule on vibration / temperature / current` -> `xgboost on raw features`: +0.1653 val PR-AUC -- **gain**
- `TFT on engineered channels` -> `ensemble lstm_engineered + tft_engineered`: +0.0272 val PR-AUC -- **gain**
- `LSTM on engineered channels` -> `TFT on engineered channels`: +0.0041 val PR-AUC -- no change
- `xgboost on engineered features` -> `LSTM on raw channels`: -0.1432 val PR-AUC -- regression

**Kept:** xgboost on raw features, xgboost on engineered features, LSTM on engineered channels, ensemble lstm_engineered + tft_engineered  
**Removed:** LSTM on raw channels, TFT on engineered channels

**Best test F1:** `ensemble lstm_engineered + tft_engineered` at 0.5142 (+0.2572 over the baseline's 0.2570).

_Total experiment compute: 11.3 min across 7 recorded runs._