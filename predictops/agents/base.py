"""Agent base class.

Every agent follows the same contract:

1.  `investigate()` computes structured findings with deterministic tools.
    This is where all numbers come from.
2.  `narrate()` optionally asks the LLM to rank hypotheses and phrase the
    result, given only the findings from step 1.  If there is no provider, or
    the provider misbehaves, the deterministic findings are used unchanged.
3.  `execute()` wraps both, times them, and writes a trajectory row.

An agent may not put a number in its output that did not come out of step 1.
`agents.verifier` re-derives the claims from raw data and flags any that do not
reconcile, which is what stops the narrative layer from inventing evidence.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..experiments.registry import ExperimentStore
from ..llm.provider import LLMProvider, LLMResult, MockProvider, UsageMeter
from ..ml.dataset import PreparedData


@dataclass
class AgentContext:
    """Structured state passed along the workflow.

    Agents exchange JSON-shaped dicts through `state`, never free text blobs.
    """

    run_id: str
    data: PreparedData
    store: ExperimentStore
    provider: LLMProvider = field(default_factory=MockProvider)
    usage: UsageMeter = field(default_factory=UsageMeter)
    state: dict = field(default_factory=dict)
    step: int = 0
    verbose: bool = True

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def say(self, msg: str) -> None:
        if self.verbose:
            print(msg)


@dataclass
class AgentResult:
    agent: str
    action: str
    reason: str
    output: dict
    tools_used: list[str] = field(default_factory=list)
    verification: str = ""
    retry_count: int = 0
    duration_s: float = 0.0
    llm: dict | None = None

    def to_dict(self) -> dict:
        d = {"agent": self.agent, "action": self.action, "reason": self.reason,
             "tools_used": self.tools_used, "verification": self.verification,
             "retry_count": self.retry_count,
             "duration_s": round(self.duration_s, 3), "output": self.output}
        if self.llm:
            d["llm"] = self.llm
        return d


class Agent(ABC):
    name: str = "agent"
    #: A one-line statement of what this agent is responsible for.  It is the
    #: agent's instruction, and it is written to the trajectory so a reader can
    #: see the brief alongside the behaviour.
    brief: str = ""
    system_prompt: str = ""

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries

    # -- the two halves ----------------------------------------------------
    @abstractmethod
    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        """Deterministic tool work.  Must return a JSON-shaped dict."""

    def narrate(self, ctx: AgentContext, findings: dict) -> tuple[dict, LLMResult | None]:
        """Optional LLM pass.  Default: none."""
        return findings, None

    def tools(self) -> list[str]:
        return []

    # -- driver -------------------------------------------------------------
    def execute(self, ctx: AgentContext, **kwargs) -> AgentResult:
        t0 = time.time()
        retries, findings, last_error = 0, None, ""
        while findings is None:
            try:
                findings = self.investigate(ctx, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                retries += 1
                if retries > self.max_retries:
                    findings = {"error": last_error, "status": "failed"}
                    break

        llm_meta = None
        if findings.get("status") != "failed":
            findings, llm_result = self.narrate(ctx, findings)
            if llm_result is not None:
                ctx.usage.add(self.name, llm_result)
                llm_meta = llm_result.to_dict()

        result = AgentResult(
            agent=self.name,
            action=findings.get("_action", self.brief or self.name),
            reason=findings.get("_reason", ""),
            output={k: v for k, v in findings.items() if not k.startswith("_")},
            tools_used=self.tools(),
            verification=findings.get("_verification", ""),
            retry_count=retries,
            duration_s=time.time() - t0,
            llm=llm_meta,
        )
        ctx.store.log_step(
            run_id=ctx.run_id, agent=self.name, step=ctx.next_step(),
            action=result.action, reason=result.reason,
            tools_used=result.tools_used,
            input_summary=self._summarise_input(kwargs),
            output=result.output, verification=result.verification,
            retry_count=result.retry_count, duration_s=result.duration_s)
        return result

    @staticmethod
    def _summarise_input(kwargs: dict) -> str:
        parts = []
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool)):
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}=<{type(v).__name__}>")
        return ", ".join(parts)
