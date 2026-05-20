import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import settings
from core.memory_manager import HierarchicalMemoryManager
from core.retriever import HybridRetriever
from core.safety_guardrail import TCMMedicalSafetyGuardrail


class AgentAction(str, Enum):
    RETRIEVE = "retrieve"
    REWRITE_QUERY = "rewrite_query"
    GENERATE = "generate"
    CLARIFY = "clarify"


class AgentState(TypedDict):
    user_query: str
    rewritten_query: str
    retrieved_evidence: List[Dict[str, Any]]
    all_evidence_pool: List[Dict[str, Any]]
    retrieval_iterations: int
    cot_reasoning: str
    evidence_consistency_score: float
    confidence_score: float
    next_action: str
    memory_context: str
    final_answer: str
    citations: List[str]
    safety_action: str
    safety_risk_level: str
    safety_reasons: List[str]
    safety_missing_info: List[str]
    safety_flags: List[str]


@dataclass
class CoTReasoningResult:
    reasoning: str = ""
    next_action: AgentAction = AgentAction.RETRIEVE
    confidence: float = 0.0
    evidence_consistency: float = 0.0
    rewritten_query: str = ""
    action_rationale: str = ""


class LLMClient:
    """LLM 调用封装，支持文本与 JSON 输出。"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
        )
        self.llm_json = self.llm.bind(response_format={"type": "json_object"})

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return str(response.content)

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        response = self.llm_json.invoke([HumanMessage(content=prompt)])
        try:
            return json.loads(str(response.content))
        except json.JSONDecodeError:
            logger.error(f"JSON 解析失败: {response.content}")
            return {}


class TCMReActAgent:
    """
    中医问答 ReAct Agent
    """

    def __init__(self, retriever: HybridRetriever, memory_manager: HierarchicalMemoryManager):
        self.retriever = retriever
        self.memory_manager = memory_manager
        self.llm_client = LLMClient()
        self.guardrail = TCMMedicalSafetyGuardrail()

    def _format_evidence_for_prompt(self, evidence_pool: List[Dict[str, Any]]) -> str:
        if not evidence_pool:
            return "（暂无证据）"

        formatted = []
        for i, evidence in enumerate(evidence_pool, 1):
            formatted.append(
                f"【证据{i}】（相关性评分: {evidence.get('rerank_score', 0):.3f}）\n"
                f"来源: {evidence.get('citation', '未知')}\n"
                f"内容: {evidence['content'][:400]}"
                f"{'...' if len(evidence['content']) > 400 else ''}"
            )
        return "\n\n".join(formatted)

    def _merge_evidence(
        self,
        existing_pool: List[Dict[str, Any]],
        chunks,
    ) -> List[Dict[str, Any]]:
        new_evidence = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "rerank_score": chunk.rerank_score,
                "citation": chunk.citation,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

        existing_ids = {item["chunk_id"] for item in existing_pool}
        unique_new = [item for item in new_evidence if item["chunk_id"] not in existing_ids]
        return existing_pool + unique_new

    def _heuristic_reasoning(
        self,
        query: str,
        evidence_pool: List[Dict[str, Any]],
        safety_action: str,
        retrieval_iterations: int,
    ) -> CoTReasoningResult:
        evidence_count = len(evidence_pool)
        if safety_action == "clarify":
            return CoTReasoningResult(
                reasoning="安全层判断当前问题缺少辨证信息，优先追问。",
                next_action=AgentAction.CLARIFY,
                confidence=0.35,
                evidence_consistency=0.3,
                action_rationale="缺少症状细节，先追问。",
            )

        if evidence_count == 0 and retrieval_iterations >= settings.MAX_REACT_ITERATIONS:
            return CoTReasoningResult(
                reasoning="多轮检索后仍无有效证据，进入追问。",
                next_action=AgentAction.CLARIFY,
                confidence=0.2,
                evidence_consistency=0.2,
                action_rationale="证据不足。",
            )

        if evidence_count <= 1:
            return CoTReasoningResult(
                reasoning="当前证据较少，继续检索更稳妥。",
                next_action=AgentAction.RETRIEVE,
                confidence=0.45,
                evidence_consistency=0.5,
                action_rationale="证据数量不足。",
            )

        return CoTReasoningResult(
            reasoning="当前证据已具备回答基础，可以尝试生成答案。",
            next_action=AgentAction.GENERATE,
            confidence=0.78,
            evidence_consistency=0.72,
            action_rationale="证据数量与相关性已达阈值。",
        )

    def _cot_reasoning(
        self,
        query: str,
        evidence_pool: List[Dict[str, Any]],
        memory_context: str,
        retrieval_iterations: int,
        safety_action: str,
    ) -> CoTReasoningResult:
        evidence_text = self._format_evidence_for_prompt(evidence_pool)
        cot_prompt = f"""你是一位资深中医问答助手，请基于已有证据判断下一步动作。

患者历史上下文：
{memory_context or "（暂无历史记录）"}

当前问题：
{query}

已检索证据（第 {retrieval_iterations} 轮）：
{evidence_text}

请输出严格 JSON：
{{
  "reasoning": "逐步分析",
  "evidence_consistency_score": 0.0,
  "confidence_score": 0.0,
  "next_action": "retrieve|rewrite_query|generate|clarify",
  "rewritten_query": "",
  "action_rationale": "简短原因"
}}"""

        try:
            result_json = self.llm_client.generate_json(cot_prompt)
            next_action = result_json.get("next_action", "retrieve")
            return CoTReasoningResult(
                reasoning=result_json.get("reasoning", ""),
                next_action=AgentAction(next_action)
                if next_action in AgentAction._value2member_map_
                else AgentAction.RETRIEVE,
                confidence=float(result_json.get("confidence_score", 0.5)),
                evidence_consistency=float(
                    result_json.get("evidence_consistency_score", 0.5)
                ),
                rewritten_query=result_json.get("rewritten_query", ""),
                action_rationale=result_json.get("action_rationale", ""),
            )
        except Exception as e:
            logger.error(f"CoT 推理失败，使用启发式策略: {e}")
            return self._heuristic_reasoning(
                query=query,
                evidence_pool=evidence_pool,
                safety_action=safety_action,
                retrieval_iterations=retrieval_iterations,
            )

    def _generate_answer(
        self,
        query: str,
        evidence_pool: List[Dict[str, Any]],
        memory_context: str,
        cot_reasoning: str,
        confidence: float,
        consistency: float,
    ) -> Dict[str, Any]:
        evidence_text = self._format_evidence_for_prompt(evidence_pool)
        evidence_count = len(evidence_pool)

        if confidence < 0.5 or evidence_count <= 1:
            answer_mode = "low_confidence"
            tone_instruction = "语气需明显保守，不要下定论。"
        elif confidence < settings.CONFIDENCE_THRESHOLD or consistency < settings.EVIDENCE_CONSISTENCY_THRESHOLD:
            answer_mode = "moderate_confidence"
            tone_instruction = "保持谨慎，说明仍需更多症状、舌脉或病程信息。"
        else:
            answer_mode = "supported"
            tone_instruction = "可以给出较完整分析，但仍应避免过度确定。"

        generation_prompt = f"""你是一位谨慎、专业的中医问答助手，请用中文回答患者。

模式：{answer_mode}
当前信号：
- confidence: {confidence:.3f}
- consistency: {consistency:.3f}
- evidence_count: {evidence_count}

写作要求：
1. 严格基于检索证据回答。
2. 使用谨慎措辞，如“倾向于”“可能”“仍需结合舌脉进一步判断”。
3. 不要给出精确剂量。
4. 如提及方剂、中药或调理建议，只能作为参考方向。
5. 单独增加“证据边界”一节，区分直接证据与合理推断。
6. 不替代线下诊断。

语气要求：
{tone_instruction}

患者历史上下文：
{memory_context or "None"}

患者问题：
{query}

检索证据：
{evidence_text}

推理摘要：
{cot_reasoning[:500] if cot_reasoning else "None"}
"""

        logger.info("[node:generate_answer] 开始生成最终答案")
        answer = self.llm_client.generate(generation_prompt)
        answer, safety_flags = self.guardrail.sanitize_answer(answer)

        citations = [item.get("citation", "") for item in evidence_pool if item.get("citation")]
        unique_citations = list(dict.fromkeys(citations))

        return {
            "answer": answer,
            "citations": unique_citations,
            "safety_flags": safety_flags,
        }

    def _clarify(
        self,
        query: str,
        confidence: float,
        evidence_pool: List[Dict[str, Any]],
        decision,
    ) -> Dict[str, Any]:
        clarification = self.guardrail.build_clarification_response(
            query=query,
            decision=decision,
            confidence=confidence,
            evidence_count=len(evidence_pool),
        )
        return {
            "answer": clarification,
            "citations": [],
            "safety_flags": [],
        }

    def run(self, user_query: str, update_memory: bool = True) -> Dict[str, Any]:
        logger.info(f"=== TCM ReAct Agent 启动 === Query: {user_query}")

        safety_decision = self.guardrail.assess_query(user_query)
        if safety_decision.action == "urgent_referral":
            answer = self.guardrail.build_urgent_response(safety_decision)
            if update_memory:
                self.memory_manager.update("user", user_query)
                self.memory_manager.update("assistant", answer)
            return {
                "answer": answer,
                "citations": [],
                "cot_reasoning": "",
                "confidence": 0.0,
                "retrieval_iterations": 0,
                "all_evidence_pool": [],
                "evidence_count": 0,
                "safety_action": safety_decision.action,
                "safety_risk_level": safety_decision.risk_level,
                "safety_reasons": safety_decision.reasons,
                "safety_missing_info": safety_decision.missing_info,
                "safety_flags": ["urgent_referral"],
            }

        memory_context = self.memory_manager.get_relevant_context(user_query)
        evidence_pool: List[Dict[str, Any]] = []
        cot_result = CoTReasoningResult()

        for iteration in range(1, settings.MAX_REACT_ITERATIONS + 1):
            search_query = cot_result.rewritten_query or user_query
            logger.info(f"[检索] 第 {iteration} 轮，Query: {search_query}")
            chunks = self.retriever.retrieve(search_query)
            evidence_pool = self._merge_evidence(evidence_pool, chunks)

            cot_result = self._cot_reasoning(
                query=user_query,
                evidence_pool=evidence_pool,
                memory_context=memory_context,
                retrieval_iterations=iteration,
                safety_action=safety_decision.action,
            )

            if (
                cot_result.confidence >= settings.CONFIDENCE_THRESHOLD
                and cot_result.evidence_consistency >= settings.EVIDENCE_CONSISTENCY_THRESHOLD
            ):
                cot_result.next_action = AgentAction.GENERATE

            if cot_result.next_action == AgentAction.GENERATE:
                break
            if cot_result.next_action == AgentAction.CLARIFY:
                break

        if cot_result.next_action == AgentAction.CLARIFY:
            generated = self._clarify(
                query=user_query,
                confidence=cot_result.confidence,
                evidence_pool=evidence_pool,
                decision=safety_decision,
            )
            final_answer = generated["answer"]
            citations = generated["citations"]
            safety_flags = generated["safety_flags"]
            safety_action = "clarify"
        else:
            generated = self._generate_answer(
                query=user_query,
                evidence_pool=evidence_pool,
                memory_context=memory_context,
                cot_reasoning=cot_result.reasoning,
                confidence=cot_result.confidence,
                consistency=cot_result.evidence_consistency,
            )
            final_answer = generated["answer"]
            citations = generated["citations"]
            safety_flags = generated["safety_flags"]
            safety_action = safety_decision.action

        if update_memory:
            self.memory_manager.update("user", user_query)
            self.memory_manager.update("assistant", final_answer)

        result = {
            "answer": final_answer,
            "citations": citations,
            "cot_reasoning": cot_result.reasoning,
            "confidence": cot_result.confidence,
            "retrieval_iterations": min(
                settings.MAX_REACT_ITERATIONS,
                max(1, len(evidence_pool) and 1 or 0),
            ),
            "all_evidence_pool": evidence_pool,
            "evidence_count": len(evidence_pool),
            "safety_action": safety_action,
            "safety_risk_level": safety_decision.risk_level,
            "safety_reasons": safety_decision.reasons,
            "safety_missing_info": safety_decision.missing_info,
            "safety_flags": safety_flags,
        }

        logger.info(
            f"=== 问答完成 === 迭代次数: {result['retrieval_iterations']}, "
            f"置信度: {result['confidence']:.3f}, 引用来源: {len(result['citations'])}"
        )
        return result
