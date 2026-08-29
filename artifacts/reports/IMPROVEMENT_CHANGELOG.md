## Improvement Changelog

Run `main`. Every figure below was read back from the experiment registry (`artifacts/experiments/experiments.db`); none is hand-entered.

Models are **selected on validation PR-AUC**. Test F1 is reported for the same frozen threshold and is never used to choose between candidates.

| Stage | What was tried and why | Val PR-AUC | Test F1 | vs baseline | Decision / learning |
|---|---|---|---|---|---|
| **Baseline** | threshold rule on vibration / temperature / current<br/><sub>A per-machine z-score alarm is what the plant already runs; establish where that leaves us.</sub> | 0.1517 | 0.2588 | -- | `reference` Establishes the operating point the plant already has. |
| **Iteration 1** | xgboost on raw features<br/><sub>Do gradient-boosted trees over raw features beat the alarm rule without any sequence modelling?</sub> | 0.4285 | 0.4625 | +0.2038 | `kept` PR-AUC 0.4285 vs baseline 0.1517 (+0.2768). Trees on raw channels already beat the alarm rule, so most of the baseline's weakness was the fixed threshold, not the sensors. |
| **Iteration 2** | xgboost on engineered features<br/><sub>Do gradient-boosted trees over engineered features beat the alarm rule without any sequence modelling?</sub> | 0.6112 | 0.5364 | +0.2776 | `kept` PR-AUC 0.6112 vs raw features 0.4285 (+0.1827). Causal rolling statistics and load/ambient normalisation are worth more than any change of architecture tried below. |
| **Iteration 3** | LSTM on raw channels<br/><sub>Does an LSTM over a 36-step window on raw channels capture degradation the tabular view misses?</sub> | 0.4346 | 0.3778 | +0.1190 | `removed` PR-AUC 0.4346 vs best tabular 0.6112 (-0.1766). Learning the temporal structure from raw sequence does not beat handing a tree the same structure explicitly. |
| **Iteration 4** | LSTM on engineered channels<br/><sub>Does an LSTM over a 36-step window on engineered channels capture degradation the tabular view misses?</sub> | 0.5844 | 0.3616 | +0.1028 | `removed` PR-AUC 0.5844 vs best tabular 0.6112 (-0.0268). Learning the temporal structure from raw sequence does not beat handing a tree the same structure explicitly. |
| **Iteration 5** | TFT on engineered channels<br/><sub>Does variable selection plus attention beat a plain LSTM on the same channels?</sub> | 0.5756 | 0.4977 | +0.2389 | `removed` PR-AUC 0.5756 vs LSTM 0.5844. No material gain over the LSTM at this data scale -- the extra capacity is not paying for itself. |
| **Iteration 6** | ensemble lstm_engineered + tft_engineered<br/><sub>Do the two sequence models make different mistakes? If so a validation-weighted blend should beat either alone.</sub> | 0.6167 | 0.4627 | +0.2039 | `kept` PR-AUC 0.6167 vs best member 0.5844. The blend adds real signal. |

### Operational view

Row-level F1 is the model-selection metric. What a maintenance planner actually feels is the next table: how many real failures were caught, how much warning they got, and how many times the crew was sent out for nothing.

| Stage | Model | Events caught | Mean early warning (h) | False alarms / machine / day | Precision | Recall |
|---|---|---|---|---|---|---|
| Baseline | threshold_baseline | 19/26 (73%) | 4.11 | 1.750 | 0.226 | 0.303 |
| Iteration 1 | xgboost | 21/26 (81%) | 3.94 | 0.489 | 0.572 | 0.388 |
| Iteration 2 | xgboost | 23/26 (88%) | 4.01 | 0.514 | 0.611 | 0.478 |
| Iteration 3 | lstm | 11/26 (42%) | 3.71 | 0.264 | 0.633 | 0.269 |
| Iteration 4 | lstm | 14/26 (54%) | 2.81 | 0.364 | 0.554 | 0.268 |
| Iteration 5 | tft | 17/26 (65%) | 4.64 | 0.500 | 0.591 | 0.430 |
| Iteration 6 | ensemble | 17/26 (65%) | 3.61 | 0.360 | 0.631 | 0.365 |

### Where the improvement came from

- `threshold rule on vibration / temperature / current` -> `xgboost on raw features`: +0.2768 val PR-AUC -- **gain**
- `xgboost on raw features` -> `xgboost on engineered features`: +0.1827 val PR-AUC -- **gain**
- `LSTM on raw channels` -> `LSTM on engineered channels`: +0.1498 val PR-AUC -- **gain**
- `TFT on engineered channels` -> `ensemble lstm_engineered + tft_engineered`: +0.0410 val PR-AUC -- **gain**
- `LSTM on engineered channels` -> `TFT on engineered channels`: -0.0087 val PR-AUC -- no change
- `xgboost on engineered features` -> `LSTM on raw channels`: -0.1766 val PR-AUC -- regression

**Kept:** xgboost on raw features, xgboost on engineered features, ensemble lstm_engineered + tft_engineered  
**Removed:** LSTM on raw channels, LSTM on engineered channels, TFT on engineered channels

**Best test F1:** `xgboost on engineered features` at 0.5364 (+0.2776 over the baseline's 0.2588).

_Total experiment compute: 17.6 min across 7 recorded runs._