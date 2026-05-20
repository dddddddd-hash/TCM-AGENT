from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from core.safety_guardrail import TCMMedicalSafetyGuardrail


@dataclass
class SafetyEvalResult:
    total: int
    passed: int
    failed: int
    accuracy: float
    failures: List[Dict[str, Any]]
    by_category: Dict[str, Dict[str, float]]


class SafetyEvalRunner:
    def __init__(self, dataset_path: str = "eval/safety_eval_set.json"):
        self.dataset_path = Path(dataset_path)
        self.guardrail = TCMMedicalSafetyGuardrail()

    def run(self) -> SafetyEvalResult:
        cases = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        failures: List[Dict[str, Any]] = []
        by_category: Dict[str, Dict[str, float]] = {}

        for case in cases:
            category = case.get("category", "default")
            if category not in by_category:
                by_category[category] = {"total": 0, "passed": 0, "failed": 0, "accuracy": 0.0}
            by_category[category]["total"] += 1

            decision = self.guardrail.assess_query(case["query"])
            expected = case["expected_action"]
            if decision.action != expected:
                by_category[category]["failed"] += 1
                failures.append(
                    {
                        "id": case["id"],
                        "category": category,
                        "query": case["query"],
                        "expected": expected,
                        "actual": decision.action,
                        "risk_level": decision.risk_level,
                        "reasons": decision.reasons,
                    }
                )
            else:
                by_category[category]["passed"] += 1

        total = len(cases)
        failed = len(failures)
        passed = total - failed
        accuracy = passed / total if total else 0.0
        for stats in by_category.values():
            stats["accuracy"] = stats["passed"] / stats["total"] if stats["total"] else 0.0
        return SafetyEvalResult(
            total=total,
            passed=passed,
            failed=failed,
            accuracy=accuracy,
            failures=failures,
            by_category=by_category,
        )
