import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import settings


@dataclass
class EvaluationRecord:
    """单条问答评估记录。"""

    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    faithfulness_score: float = 0.0
    answer_relevancy_score: float = 0.0
    context_recall_score: float = 0.0


class RAGASEvaluator:
    """DeepSeek 兼容评估器。"""

    def __init__(self, preloaded_embed_model: Any = None):
        self.eval_chat_llm = ChatOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL_NAME,
            temperature=0.0,
        )
        self.enable_answer_relevancy = True

    @staticmethod
    def _clamp_score(score: Any) -> float:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found: {text[:100]}")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _content_chars(text: str) -> List[str]:
        return [
            c
            for c in text
            if c.isalnum() or (ord(c) > 127 and not c.isspace())
        ]

    def _fallback_faithfulness(self, answer: str, contexts: List[str]) -> float:
        if not answer.strip() or not contexts:
            return 0.0
        answer_chars = set(self._content_chars(answer))
        context_chars = set(self._content_chars("\n".join(contexts)))
        if not answer_chars or not context_chars:
            return 0.0
        return self._clamp_score(len(answer_chars & context_chars) / len(answer_chars))

    def _fallback_answer_relevancy(self, question: str, answer: str) -> float:
        if not question.strip() or not answer.strip():
            return 0.0
        stop_chars = set("的了呢吗么什是和与及请问一个哪些如何怎样")
        question_chars = {c for c in self._content_chars(question) if c not in stop_chars}
        answer_chars = set(self._content_chars(answer))
        if not question_chars or not answer_chars:
            return 0.0
        overlap = len(question_chars & answer_chars) / len(question_chars)
        return self._clamp_score(0.35 + 0.65 * overlap)

    def _score_with_deepseek(self, prompt: str, fallback_score: float) -> float:
        try:
            response = self.eval_chat_llm.invoke([HumanMessage(content=prompt)])
            payload = self._extract_json(str(response.content).strip())
            return self._clamp_score(payload.get("score", fallback_score))
        except Exception as e:
            logger.warning(f"DeepSeek 评估失败，使用兜底分数: {e}")
            return fallback_score

    def _score_faithfulness(self, answer: str, contexts: List[str]) -> float:
        fallback = self._fallback_faithfulness(answer, contexts)
        context_text = "\n\n".join(contexts[:8])
        prompt = f"""你是一个严格的 RAG 忠实度评估器。请判断答案中的关键事实是否被参考证据支持。

评分规则：
1. 只看答案是否忠实于证据，不评估文笔。
2. 如果答案的大部分关键事实都能在证据中找到支持，给 0.8-1.0。
3. 如果答案部分有依据但也有明显未支持内容，给 0.4-0.7。
4. 如果证据为空或答案主要无法被证据支持，给 0-0.3。

只输出 JSON：{{"score": 0.0, "reason": "简短原因"}}

参考证据：
{context_text}

答案：
{answer}
"""
        return self._score_with_deepseek(prompt, fallback)

    def _score_answer_relevancy(self, question: str, answer: str) -> float:
        fallback = self._fallback_answer_relevancy(question, answer)
        prompt = f"""你是一个严格的问答相关性评估器。请判断答案是否直接、完整地回答了问题。

评分规则：
1. 只评估答案是否切题，不评估事实正确性。
2. 直接且完整回答问题，给 0.8-1.0。
3. 部分回答或有明显偏题，给 0.4-0.7。
4. 基本没有回答问题，给 0-0.3。

只输出 JSON：{{"score": 0.0, "reason": "简短原因"}}

问题：
{question}

答案：
{answer}
"""
        return self._score_with_deepseek(prompt, fallback)

    def evaluate_batch(self, records: List[EvaluationRecord]) -> Dict[str, float]:
        """批量评估问答记录并返回平均分。"""
        if not records:
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}

        faithfulness_scores = [
            self._score_faithfulness(r.answer, r.contexts) for r in records
        ]
        answer_relevancy_scores = [
            self._score_answer_relevancy(r.question, r.answer) for r in records
        ]

        scores = {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
            "answer_relevancy": sum(answer_relevancy_scores) / len(answer_relevancy_scores),
            "context_recall": 0.0,
        }
        logger.info(f"自动评估完成: {scores}")
        return scores

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationRecord:
        """评估单条问答记录。"""
        record = EvaluationRecord(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        scores = self.evaluate_batch([record])
        record.faithfulness_score = scores.get("faithfulness", 0.0)
        record.answer_relevancy_score = scores.get("answer_relevancy", 0.0)
        record.context_recall_score = scores.get("context_recall", 0.0)
        return record

    def generate_evaluation_report(
        self, records: List[EvaluationRecord], output_path: str = "eval_report.json"
    ) -> Dict[str, Any]:
        """生成评估报告。"""
        scores = self.evaluate_batch(records)
        low_faithfulness = [r for r in records if r.faithfulness_score < 0.5]

        report = {
            "summary": {
                "total_samples": len(records),
                "avg_faithfulness": scores.get("faithfulness", 0.0),
                "avg_answer_relevancy": scores.get("answer_relevancy", 0.0),
                "avg_context_recall": scores.get("context_recall", 0.0),
            },
            "risk_cases": [
                {
                    "question": r.question,
                    "answer_snippet": r.answer[:100],
                    "faithfulness": r.faithfulness_score,
                    "risk_level": "HIGH" if r.faithfulness_score < 0.3 else "MEDIUM",
                }
                for r in low_faithfulness
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"评估报告已保存: {output_path}")
        return report
