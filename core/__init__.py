from core.retriever import HybridRetriever, RetrievedChunk, FAISSHNSWIndex
from core.memory_manager import (
    HierarchicalMemoryManager,
    MedicalNERTagger,
    NLIConflictDetector,
    LongTermMemoryStore,
    ShortTermMemoryManager,
    MemoryEntry,
)
from core.agent_workflow import TCMReActAgent, AgentState, LLMClient
from core.evaluator import RAGASEvaluator, EvaluationRecord
from core.safety_guardrail import SafetyDecision, TCMMedicalSafetyGuardrail

import os
from typing import Optional
from loguru import logger


class SystemFactory:
    """系统组装工厂。"""

    @staticmethod
    def build(
        index_path: Optional[str] = None,
        load_existing_index: bool = True,
    ):
        """构建 agent、retriever 和 memory manager。"""
        from config.settings import settings

        resolve_index_path = index_path or settings.FAISS_INDEX_PATH

        logger.info("◆" * 50)
        logger.info("  TCM-RAG 系统初始化中...")
        logger.info("◆" * 50)

        logger.info("[1/3] 初始化 HybridRetriever（HNSW + BM25 + Rerank）...")
        retriever = HybridRetriever()

        if load_existing_index and os.path.exists(resolve_index_path):
            logger.info(f"      检测到已有索引，直接加载: {resolve_index_path}")
            retriever.hnsw_index.load(resolve_index_path)
        else:
            logger.warning(
                f"      未找到索引文件: {resolve_index_path}\n"
                f"      请调用 retriever.build_index(chunks) 后再使用检索功能"
            )

        logger.info("[2/3] 初始化 LLMClient 与 HierarchicalMemoryManager...")
        llm_client = LLMClient()
        memory_manager = HierarchicalMemoryManager(llm_client)

        logger.info("[3/3] 组装 TCMReActAgent（CoT + ReAct 工作流）...")
        agent = TCMReActAgent(retriever, memory_manager)

        logger.info("◆" * 50)
        logger.info("  ✓ 系统初始化完成，可以开始问答")
        logger.info("◆" * 50)

        return agent, retriever, memory_manager

    @staticmethod
    def build_evaluator() -> RAGASEvaluator:
        logger.info("初始化 RAGAS 评估器...")
        return RAGASEvaluator()


__all__ = [
    "HybridRetriever",
    "RetrievedChunk",
    "FAISSHNSWIndex",
    "HierarchicalMemoryManager",
    "MedicalNERTagger",
    "NLIConflictDetector",
    "LongTermMemoryStore",
    "ShortTermMemoryManager",
    "MemoryEntry",
    "TCMReActAgent",
    "AgentState",
    "LLMClient",
    "SafetyDecision",
    "TCMMedicalSafetyGuardrail",
    "RAGASEvaluator",
    "EvaluationRecord",
    "SystemFactory",
]
