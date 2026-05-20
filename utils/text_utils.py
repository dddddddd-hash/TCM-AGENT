import re
import json
import hashlib
import os
from typing import List, Dict, Any
from loguru import logger


class SemanticChunker:
    """面向检索的轻量语义切分器。"""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        min_chunk_size: int = 100,
        cache_dir: str = "data/processed",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _split_sentences(self, text: str) -> List[str]:
        """按中文句末标点切分文本。"""
        text = re.sub(r"([。！？；])", r"\1<SPLIT>", text)
        text = re.sub(r"\n", "<SPLIT>", text)
        sentences = [s.strip() for s in text.split("<SPLIT>") if s.strip()]
        return sentences

    def _merge_sentences(
        self,
        sentences: List[str],
        metadata: Dict,
    ) -> List[Dict[str, Any]]:
        """将句子合并为固定长度附近的文本块。"""
        chunks = []
        current = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            if current_len + sent_len > self.chunk_size and current:
                chunk_text = "".join(current)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(
                        {
                            "content": chunk_text,
                            "metadata": {**metadata, "chunk_index": len(chunks)},
                        }
                    )

                overlap_text = chunk_text[-self.chunk_overlap:]
                current = [overlap_text]
                current_len = len(overlap_text)

            current.append(sent)
            current_len += sent_len

        if current:
            chunk_text = "".join(current)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {**metadata, "chunk_index": len(chunks)},
                    }
                )

        return chunks

    def _get_cache_path(self, raw_content: str, metadata: Dict) -> str:
        key = hashlib.md5(
            (raw_content[:500] + str(metadata.get("source", ""))).encode()
        ).hexdigest()
        return os.path.join(self.cache_dir, f"chunk_{key}.json")

    def _load_cache(self, cache_path: str):
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, cache_path: str, chunks: List[Dict]):
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def chunk_document(
        self,
        raw_content: str,
        metadata: Dict,
    ) -> List[Dict[str, Any]]:
        """切分单篇文档。"""
        cache_path = self._get_cache_path(raw_content, metadata)
        cached = self._load_cache(cache_path)
        if cached is not None:
            logger.debug(f"命中缓存：{cache_path}")
            return cached

        sentences = self._split_sentences(raw_content)
        chunks = self._merge_sentences(sentences, metadata)

        self._save_cache(cache_path, chunks)

        logger.info(
            f"语义切分完成：第 {metadata.get('page', '?')} 页 -> {len(chunks)} 个 Chunk"
        )
        return chunks
