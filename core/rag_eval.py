from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from core.agent_workflow import TCMReActAgent
from core.evaluator import RAGASEvaluator


@dataclass
class RAGEvalSummary:
    total_samples: int
    answer_samples: int
    clarify_samples: int
    urgent_referral_samples: int
    route_accuracy: float
    clarify_success_rate: float
    proceed_success_rate: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_confidence: float
    avg_answer_confidence: float
    avg_retrieval_iterations: float
    avg_evidence_count: float
    avg_citation_count: float
    clarify_rate: float
    urgent_referral_rate: float
    avg_answer_length: float
    by_category: Dict[str, Dict[str, float]]
    report_path: str


class RAGEvalRunner:
    def __init__(
        self,
        agent: TCMReActAgent,
        dataset_path: str = "eval/rag_eval_set.json",
        report_path: str = "eval/rag_eval_report.json",
    ):
        self.agent = agent
        self.dataset_path = Path(dataset_path)
        self.report_path = Path(report_path)
        self.evaluator = RAGASEvaluator(
            preloaded_embed_model=getattr(agent.retriever, "embed_model", None)
        )

    def run(self) -> RAGEvalSummary:
        cases = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        details: List[Dict[str, Any]] = []

        for index, case in enumerate(cases, 1):
            question = case["question"]
            expected_action = self._expected_action(case)
            result = self.agent.run(question, update_memory=False)
            contexts = [e["content"] for e in result.get("all_evidence_pool", [])]
            safety_action = result.get("safety_action", "proceed")
            if safety_action == "proceed":
                record = self.evaluator.evaluate_single(
                    question=question,
                    answer=result["answer"],
                    contexts=contexts,
                    ground_truth=case.get("ground_truth"),
                )
                faithfulness = record.faithfulness_score
                answer_relevancy = record.answer_relevancy_score
            else:
                faithfulness = None
                answer_relevancy = None

            detail = {
                "index": index,
                "id": case["id"],
                "category": case.get("category", "default"),
                "question": question,
                "expected_action": expected_action,
                "route_correct": safety_action == expected_action,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "confidence": result.get("confidence", 0.0),
                "retrieval_iterations": result.get("retrieval_iterations", 0),
                "evidence_count": len(contexts),
                "citation_count": len(result.get("citations", [])),
                "answer_length": len(result.get("answer", "")),
                "safety_action": safety_action,
                "safety_risk_level": result.get("safety_risk_level", "low"),
                "safety_flags": result.get("safety_flags", []),
                "answer_snippet": result.get("answer", "")[:180],
            }
            details.append(detail)

        summary = self._summarize(details)
        report = {
            "summary": summary.__dict__,
            "details": details,
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _summarize(self, details: List[Dict[str, Any]]) -> RAGEvalSummary:
        if not details:
            return RAGEvalSummary(
                total_samples=0,
                answer_samples=0,
                clarify_samples=0,
                urgent_referral_samples=0,
                route_accuracy=0.0,
                clarify_success_rate=0.0,
                proceed_success_rate=0.0,
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_confidence=0.0,
                avg_answer_confidence=0.0,
                avg_retrieval_iterations=0.0,
                avg_evidence_count=0.0,
                avg_citation_count=0.0,
                clarify_rate=0.0,
                urgent_referral_rate=0.0,
                avg_answer_length=0.0,
                by_category={},
                report_path=str(self.report_path),
            )

        answered = [d for d in details if d["safety_action"] == "proceed"]
        expected_clarify = [d for d in details if d["expected_action"] == "clarify"]
        expected_proceed = [d for d in details if d["expected_action"] == "proceed"]

        by_category: Dict[str, Dict[str, float]] = {}
        for category in sorted({d["category"] for d in details}):
            subset = [d for d in details if d["category"] == category]
            answered_subset = [d for d in subset if d["safety_action"] == "proceed"]
            by_category[category] = {
                "total": len(subset),
                "answer_samples": len(answered_subset),
                "route_accuracy": self._ratio([d["route_correct"] for d in subset]),
                "avg_faithfulness": self._mean_metric(answered_subset, "faithfulness"),
                "avg_answer_relevancy": self._mean_metric(answered_subset, "answer_relevancy"),
                "avg_confidence": mean(d["confidence"] for d in subset),
                "avg_evidence_count": mean(d["evidence_count"] for d in subset),
                "clarify_rate": self._rate(subset, "clarify"),
            }

        return RAGEvalSummary(
            total_samples=len(details),
            answer_samples=len(answered),
            clarify_samples=sum(1 for d in details if d["safety_action"] == "clarify"),
            urgent_referral_samples=sum(1 for d in details if d["safety_action"] == "urgent_referral"),
            route_accuracy=self._ratio([d["route_correct"] for d in details]),
            clarify_success_rate=self._ratio([d["safety_action"] == "clarify" for d in expected_clarify]),
            proceed_success_rate=self._ratio([d["safety_action"] == "proceed" for d in expected_proceed]),
            avg_faithfulness=self._mean_metric(answered, "faithfulness"),
            avg_answer_relevancy=self._mean_metric(answered, "answer_relevancy"),
            avg_confidence=mean(d["confidence"] for d in details),
            avg_answer_confidence=mean(d["confidence"] for d in answered) if answered else 0.0,
            avg_retrieval_iterations=mean(d["retrieval_iterations"] for d in details),
            avg_evidence_count=mean(d["evidence_count"] for d in details),
            avg_citation_count=mean(d["citation_count"] for d in details),
            clarify_rate=self._rate(details, "clarify"),
            urgent_referral_rate=self._rate(details, "urgent_referral"),
            avg_answer_length=mean(d["answer_length"] for d in details),
            by_category=by_category,
            report_path=str(self.report_path),
        )

    @staticmethod
    def _expected_action(case: Dict[str, Any]) -> str:
        if "expected_action" in case:
            return case["expected_action"]
        if case.get("category") == "clarify":
            return "clarify"
        return "proceed"

    @staticmethod
    def _mean_metric(details: List[Dict[str, Any]], key: str) -> float:
        values = [d[key] for d in details if d.get(key) is not None]
        return mean(values) if values else 0.0

    @staticmethod
    def _ratio(values: List[bool]) -> float:
        if not values:
            return 0.0
        return sum(1 for v in values if v) / len(values)

    @staticmethod
    def _rate(details: List[Dict[str, Any]], action: str) -> float:
        if not details:
            return 0.0
        return sum(1 for d in details if d["safety_action"] == action) / len(details)
