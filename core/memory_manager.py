import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
import torch
from FlagEmbedding import FlagModel
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

from config.settings import settings


@dataclass
class MemoryEntry:
    """
    单条记忆数据结构
    """

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"
    content: str = ""
    importance: float = 1.0
    entities: List[Dict[str, str]] = field(default_factory=list)
    turn_index: int = 0
    timestamp: float = field(default_factory=time.time)
    is_summarized: bool = False
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "role": self.role,
            "content": self.content,
            "importance": self.importance,
            "entities": self.entities,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
        }


@dataclass
class ConflictCheckResult:
    has_conflict: bool = False
    conflict_pairs: List[Tuple[str, str, float]] = field(default_factory=list)


class MedicalNERTagger:
    """
    医疗命名实体识别器
    """

    FALLBACK_DICT = {
        "疾病": ["失眠", "感冒", "咳嗽", "头痛", "腹痛", "高血压", "糖尿病", "风寒"],
        "中药": ["黄芪", "当归", "川芎", "白芍", "熟地黄", "茯苓", "甘草", "人参"],
        "方剂": ["四物汤", "六味地黄丸", "补中益气汤", "逍遥散", "桂枝汤"],
        "症状": ["发热", "恶寒", "口渴", "乏力", "失眠", "心悸", "盗汗"],
        "治法": ["补气", "活血", "化瘀", "清热", "解毒", "温阳", "滋阴"],
        "穴位": ["足三里", "合谷", "内关", "三阴交", "关元", "气海"],
    }

    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.NER_MODEL_PATH)
            self.model = AutoModelForTokenClassification.from_pretrained(
                settings.NER_MODEL_PATH
            )
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("医疗 NER 模型加载成功")
        except Exception as e:
            logger.warning(f"NER 模型加载失败，使用规则兜底: {e}")
            self.ner_pipeline = None

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        if not self.ner_pipeline:
            return self._rule_based_ner(text)

        try:
            raw_entities = self.ner_pipeline(text)
            return [
                {
                    "entity_type": ent.get("entity_group", "OTHER"),
                    "text": ent.get("word", ""),
                    "score": float(ent.get("score", 0.0)),
                }
                for ent in raw_entities
            ]
        except Exception as e:
            logger.error(f"NER 推理失败: {e}")
            return []

    def _rule_based_ner(self, text: str) -> List[Dict[str, str]]:
        entities = []
        for entity_type, keywords in self.FALLBACK_DICT.items():
            for keyword in keywords:
                if keyword in text:
                    entities.append(
                        {
                            "entity_type": entity_type,
                            "text": keyword,
                            "score": 1.0,
                        }
                    )
        return entities

    def compute_ner_importance_weight(self, entities: List[Dict[str, str]]) -> float:
        if not entities:
            return settings.NER_WEIGHT_MAP["DEFAULT"]

        max_weight = settings.NER_WEIGHT_MAP["DEFAULT"]
        for ent in entities:
            entity_type = ent.get("entity_type", "DEFAULT")
            weight = settings.NER_WEIGHT_MAP.get(
                entity_type,
                settings.NER_WEIGHT_MAP["DEFAULT"],
            )
            max_weight = max(max_weight, weight)

        return max_weight


class NLIConflictDetector:
    """
    基于 NLI 的记忆冲突检测器
    """

    def __init__(self):
        try:
            self.nli_pipeline = pipeline(
                "text-classification",
                model=settings.NLI_MODEL_PATH,
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("NLI 冲突检测模型加载成功")
        except Exception as e:
            logger.warning(f"NLI 模型加载失败，跳过冲突检测: {e}")
            self.nli_pipeline = None

    def check_conflict(
        self,
        new_memory: str,
        existing_memories: List[str],
        threshold: float = settings.NLI_CONTRADICTION_THRESHOLD,
    ) -> ConflictCheckResult:
        result = ConflictCheckResult()

        if not self.nli_pipeline or not existing_memories:
            return result

        conflict_pairs = []
        for existing in existing_memories:
            try:
                nli_input = f"{existing} </s> {new_memory}"
                output = self.nli_pipeline(nli_input, truncation=True, max_length=512)

                if isinstance(output, dict):
                    output = [output]

                label_score_map = {item["label"].upper(): item["score"] for item in output}
                contradiction_score = label_score_map.get("CONTRADICTION", 0.0)

                if contradiction_score >= threshold:
                    conflict_pairs.append((existing, new_memory, contradiction_score))
                    logger.warning(
                        "[NLI 冲突检测] 发现矛盾记忆: "
                        f"score={contradiction_score:.3f}"
                    )
            except Exception as e:
                logger.error(f"NLI 推理出错: {e}")
                continue

        if conflict_pairs:
            result.has_conflict = True
            result.conflict_pairs = conflict_pairs

        return result


class LongTermMemoryStore:
    """
    长期记忆向量库
    """

    def __init__(self, embed_model: FlagModel, conflict_detector: NLIConflictDetector):
        self.embed_model = embed_model
        self.conflict_detector = conflict_detector
        self.memories: List[MemoryEntry] = []
        self.embeddings: List[np.ndarray] = []

        dim = settings.EMBEDDING_DIMENSION
        self.index = faiss.IndexHNSWFlat(dim, 16)
        self.index.hnsw.efSearch = 64

    def _is_duplicate(self, new_embedding: np.ndarray) -> bool:
        if len(self.embeddings) == 0:
            return False

        existing_matrix = np.array(self.embeddings)
        similarities = existing_matrix @ new_embedding
        max_sim = float(np.max(similarities))

        if max_sim > settings.LONG_TERM_SIMILARITY_THRESHOLD:
            logger.debug(f"[长期记忆去重] 相似度 {max_sim:.3f}，跳过写入")
            return True
        return False

    def write(self, entry: MemoryEntry) -> bool:
        if entry.importance < settings.LONG_TERM_IMPORTANCE_THRESHOLD:
            logger.debug(
                f"[长期记忆-Step1] importance={entry.importance:.3f} 低于阈值 "
                f"{settings.LONG_TERM_IMPORTANCE_THRESHOLD}，不写入"
            )
            return False

        embedding = self.embed_model.encode([entry.content], normalize_embeddings=True)[0]
        embedding = np.array(embedding)

        if self._is_duplicate(embedding):
            return False

        similar_memories = self.search(entry.content, top_k=5)
        similar_texts = [m.content for m in similar_memories]
        conflict_result = self.conflict_detector.check_conflict(
            new_memory=entry.content,
            existing_memories=similar_texts,
        )

        if conflict_result.has_conflict:
            logger.warning(
                f"[长期记忆-Step3] 检测到冲突，阻止写入，冲突对数: "
                f"{len(conflict_result.conflict_pairs)}"
            )
            return False

        entry.embedding = embedding
        self.memories.append(entry)
        self.embeddings.append(embedding)

        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        logger.info(f"[长期记忆] 写入成功: {entry.content[:50]}...")
        return True

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        if len(self.memories) == 0:
            return []

        query_embedding = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        query_vec = np.array(query_embedding).reshape(1, -1).astype(np.float32)

        actual_k = min(top_k, len(self.memories))
        distances, indices = self.index.search(query_vec, actual_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.memories):
                results.append(self.memories[idx])
        return results


class ShortTermMemoryManager:
    """
    短期记忆管理器（滑动窗口 + 摘要压缩）
    """

    def __init__(self, llm_client, long_term_store: LongTermMemoryStore):
        self.llm_client = llm_client
        self.long_term_store = long_term_store
        self.ner_tagger = MedicalNERTagger()

        self.window: List[MemoryEntry] = []
        self.summary: str = ""
        self.current_turn: int = 0

    def add_entry(self, role: str, content: str) -> MemoryEntry:
        self.current_turn += 1

        entities = self.ner_tagger.extract_entities(content)
        ner_weight = self.ner_tagger.compute_ner_importance_weight(entities)
        length_factor = min(1.0, len(content) / 100.0)
        role_weight = 1.2 if role == "assistant" else 1.0
        initial_importance = ner_weight * length_factor * role_weight

        entry = MemoryEntry(
            role=role,
            content=content,
            importance=initial_importance,
            entities=entities,
            turn_index=self.current_turn,
        )

        self._apply_importance_decay()
        self.window.append(entry)

        logger.debug(
            f"[短期记忆] 添加条目 Turn={self.current_turn}, role={role}, "
            f"importance={initial_importance:.3f}"
        )

        self._manage_window()
        return entry

    def _apply_importance_decay(self):
        for entry in self.window:
            delta_turns = self.current_turn - entry.turn_index
            if delta_turns > 0:
                decay = settings.IMPORTANCE_DECAY_FACTOR ** delta_turns
                entry.importance *= decay

    def _manage_window(self):
        while len(self.window) > settings.SHORT_TERM_WINDOW_SIZE:
            oldest_entry = self.window.pop(0)
            logger.debug(
                f"[短期记忆滑出] Turn={oldest_entry.turn_index}, "
                f"importance={oldest_entry.importance:.3f}"
            )
            self.long_term_store.write(oldest_entry)

        if len(self.window) >= settings.SUMMARY_TRIGGER_TURNS:
            self._trigger_summary_compression()

    def _trigger_summary_compression(self):
        compress_count = len(self.window) // 2
        to_compress = self.window[:compress_count]
        self.window = self.window[compress_count:]

        history_text = "\n".join(f"{entry.role}: {entry.content}" for entry in to_compress)
        summary_prompt = f"""请将以下中医问诊对话历史压缩为简洁摘要，
重点保留：患者症状、体征、体质、已提及的疾病名称、中药方剂、诊断结论。

对话历史：
{history_text}

请输出摘要（100字以内）："""

        try:
            new_summary = self.llm_client.generate(summary_prompt)
            self.summary = (
                f"{self.summary}\n{new_summary}".strip()
                if self.summary
                else new_summary
            )
            logger.info(f"[短期记忆摘要压缩] 压缩 {compress_count} 条 -> {new_summary[:60]}...")
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")

    def get_context_for_llm(self) -> str:
        context_parts = []

        if self.summary:
            context_parts.append(f"【对话历史摘要】\n{self.summary}")

        if self.window:
            recent_history = "\n".join(
                f"{entry.role}: {entry.content}"
                for entry in self.window
            )
            context_parts.append(f"【近期对话记录】\n{recent_history}")

        return "\n\n".join(context_parts)

    def flush_to_long_term(self):
        logger.info(f"[会话结束] 将 {len(self.window)} 条短期记忆写入长期记忆库")
        for entry in self.window:
            self.long_term_store.write(entry)
        self.window.clear()
        self.summary = ""


class HierarchicalMemoryManager:
    """
    分层记忆系统统一接口
    """

    def __init__(self, llm_client):
        self.embed_model = FlagModel(
            settings.EMBEDDING_MODEL_PATH,
            use_fp16=True,
        )
        self.conflict_detector = NLIConflictDetector()
        self.long_term = LongTermMemoryStore(self.embed_model, self.conflict_detector)
        self.short_term = ShortTermMemoryManager(llm_client, self.long_term)

    def update(self, role: str, content: str) -> MemoryEntry:
        return self.short_term.add_entry(role, content)

    def get_relevant_context(self, query: str) -> str:
        short_term_ctx = self.short_term.get_context_for_llm()

        lt_memories = self.long_term.search(query, top_k=3)
        lt_ctx = ""
        if lt_memories:
            lt_texts = "\n".join(f"- {m.content}" for m in lt_memories)
            lt_ctx = f"【长期记忆（历史关键信息）】\n{lt_texts}"

        return f"{short_term_ctx}\n\n{lt_ctx}".strip()

    def end_session(self):
        self.short_term.flush_to_long_term()
