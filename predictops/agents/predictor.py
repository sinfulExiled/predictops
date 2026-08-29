"""Agent -- Prediction.

Question it owns: **what is likely to happen?**

Thin by design, and thinner now that `ml.service` exists. It owns no
judgement: it asks the model service and reports the answer. The failure
*type* comes from the type classifier, the *window* from the time-to-failure
regressor, and *confidence* from the measured reliability curve -- never from
the language model, and never from this agent.

It reports two thresholds. `investigate_threshold` is the trigger for looking;
`alert_threshold` is the bar for alarming. Everything between them is what the
hypothesis advocates exist to resolve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ml.service import ModelService
from .base import Agent, AgentContext


class PredictionAgent(Agent):
    name = "predictor"
    brief = ("Ask the model service what is likely to happen to one machine "
             "at one timestamp, and report it unchanged.")

    def tools(self) -> list[str]:
        return ["model_service.predict", "model_service.raw_window"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        service: ModelService = kwargs["service"]
        machine_id: str = kwargs["machine_id"]
        timestamp = kwargs["timestamp"]

        p = service.predict(ctx.data.df, machine_id, timestamp)
        raw = service.raw_window(ctx.data.df, machine_id, timestamp)

        return {
            "_action": f"Scored {machine_id} at {timestamp}.",
            "_reason": (
                f"failure probability {p.failure_probability:.3f} against an "
                f"alert threshold of {p.alert_threshold:.3f} "
                f"(investigate from {p.investigate_threshold:.3f}) "
                f"-> {p.risk_band}"),
            "_verification": ("Probability, type, ETA and confidence are all "
                              "fitted-model outputs."),
            "machine_id": machine_id,
            "timestamp": str(timestamp),
            "failure_probability": p.failure_probability,
            "alert": p.alert,
            "investigate": p.investigate,
            "risk_band": p.risk_band,
            "threshold": p.alert_threshold,
            "alert_threshold": p.alert_threshold,
            "investigate_threshold": p.investigate_threshold,
            "prediction_window_hours": p.eta,
            "failure_type": p.failure_type,
            "failure_type_confidence": p.failure_type_confidence,
            "failure_type_alternatives": p.failure_type_alternatives,
            "confidence": p.confidence,
            "confidence_basis": (
                "measured share of validation cases scoring at least this "
                "high that were genuinely followed by a failure"),
            "model": p.model,
            "observed_tail": {
                c: round(float(raw[c].tail(6).mean()), 3)
                for c in ("temperature", "vibration", "current", "pressure",
                          "load")
                if c in raw.columns},
        }
