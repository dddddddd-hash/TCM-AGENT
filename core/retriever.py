import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from loguru import logger
from FlagEmbedding import FlagAutoModel, FlagReranker

from config.settings import settings


@dataclass
class RetrievedChunk:
    """
    检索结果数据类
    携带 chunk_id 以支持最终答案中的 Citation 溯源
    """

    content: str
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    chunk_id: str = ""

    @property
    def citation(self) -> str:
        meta = self.metadata
        return (
            f"[来源: {meta.get('source', '未知')} "
            f"页码: {meta.get('page', 'N/A')} "
            f"章节: {meta.get('title', 'N/A')}]"
        )


class FAISSHNSWIndex:
    """
    FAISS-HNSW 向量索引
    """

    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index = None
        self.id_to_chunk: Dict[int, Dict[str, Any]] = {}
        self._build_index()

    def _build_index(self):
        self.index = faiss.IndexHNSWFlat(self.dimension, settings.HNSW_M)
        self.index.hnsw.efConstruction = settings.HNSW_EF_CONSTRUCTION
        self.index.hnsw.efSearch = settings.HNSW_EF_SEARCH
        logger.info(
            f"FAISS-HNSW 索引已构建: dim={self.dimension}, M={settings.HNSW_M}"
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks 数量与 Embeddings 数量不匹配")

        start_id = len(self.id_to_chunk)
        vectors = embeddings.astype(np.float32)
        self.index.add(vectors)

        for i, chunk in enumerate(chunks):
            self.id_to_chunk[start_id + i] = chunk

        logger.info(
            f"已向 HNSW 索引插入 {len(chunks)} 条记录，总量: {len(self.id_to_chunk)}"
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        query = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                score = float(1.0 / (1.0 + dist))
                results.append((int(idx), score))
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.id_to_chunk, f)
        logger.info(f"HNSW 索引已保存: {path}")

    def load(self, path: str):
        self.index = faiss.read_index(path)
        with open(path + ".meta", "rb") as f:
            self.id_to_chunk = pickle.load(f)
        logger.info(f"HNSW 索引已加载: {path}，共 {len(self.id_to_chunk)} 条记录")


class HybridRetriever:
    """
    混合召回 + Rerank + 二次过滤 的完整检索链路
    """

    def __init__(self):
        self.embed_model = FlagAutoModel.from_finetuned(
            settings.EMBEDDING_MODEL_PATH,
            use_fp16=True,
            devices="cuda",
        )
        self.reranker = FlagReranker(settings.RERANK_MODEL_PATH, use_fp16=True)
        self.hnsw_index = FAISSHNSWIndex()

        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[str] = []
        self.bm25_chunk_map: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        logger.info(f"开始构建索引，共 {len(chunks)} 个 Chunk")

        contents = [chunk["content"] for chunk in chunks]

        embeddings = self.embed_model.encode(contents, batch_size=64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        embeddings = embeddings / norms
        self.hnsw_index.add_chunks(chunks, np.array(embeddings))

        self.bm25_corpus = [list(content) for content in contents]
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        self.bm25_chunk_map = chunks

        logger.info("索引构建完成（HNSW Dense + BM25 Sparse）")

    def _dense_retrieve(self, query: str, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        embedding = self.embed_model.encode([query])
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        query_embedding = (embedding / norms)[0]
        results = self.hnsw_index.search(np.array(query_embedding), top_k)

        retrieved = []
        for vec_id, score in results:
            chunk = self.hnsw_index.id_to_chunk.get(vec_id)
            if chunk:
                retrieved.append((chunk, score))
        return retrieved

    def _sparse_retrieve(self, query: str, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        if not self.bm25_index:
            return []

        query_tokens = list(query)
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.bm25_chunk_map[idx]
                norm_score = float(scores[idx]) / (float(scores[top_indices[0]]) + 1e-8)
                retrieved.append((chunk, norm_score))
        return retrieved

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Dict[str, Any], float]],
        sparse_results: List[Tuple[Dict[str, Any], float]],
        k: int = 60,
    ) -> List[Tuple[Dict[str, Any], float]]:
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            chunk_id = chunk["metadata"].get("chunk_id", str(rank))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = chunk

        for rank, (chunk, _) in enumerate(sparse_results):
            chunk_id = chunk["metadata"].get("chunk_id", str(rank))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = chunk

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], score) for cid, score in sorted_items]

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not candidates:
            return []

        pairs = [[query, chunk["content"]] for chunk, _ in candidates]
        rerank_scores = self.reranker.compute_score(pairs)

        reranked = []
        for (chunk, _), score in zip(candidates, rerank_scores):
            reranked.append((chunk, float(score)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _secondary_filter(
        self,
        reranked_results: List[Tuple[Dict[str, Any], float]],
    ) -> List[RetrievedChunk]:
        filtered: List[RetrievedChunk] = []
        seen_contents = []

        for chunk, score in reranked_results[: settings.TOP_K_RERANK * 2]:
            if score < settings.RERANK_SCORE_THRESHOLD:
                logger.debug(f"[二次过滤-Rule1] 分数 {score:.3f} 低于阈值，丢弃")
                continue

            content = chunk["content"]

            if len(content.strip()) < 20:
                logger.debug(f"[二次过滤-Rule2] 内容过短，丢弃: {content[:30]}")
                continue

            chinese_chars = sum(1 for c in content if "\u4e00" <= c <= "\u9fff")
            if len(content) > 0 and chinese_chars / len(content) < 0.3:
                logger.debug("[二次过滤-Rule3] 中文字符比例过低，疑似 OCR 噪声，丢弃")
                continue

            is_duplicate = False
            for seen in seen_contents:
                set_a, set_b = set(content), set(seen)
                jaccard = len(set_a & set_b) / (len(set_a | set_b) + 1e-8)
                if jaccard > 0.85:
                    is_duplicate = True
                    break
            if is_duplicate:
                logger.debug("[二次过滤-Rule4] 内容重复，跳过")
                continue

            seen_contents.append(content)
            filtered.append(
                RetrievedChunk(
                    content=content,
                    metadata=chunk["metadata"],
                    rerank_score=score,
                    chunk_id=chunk["metadata"].get("chunk_id", ""),
                )
            )

            if len(filtered) >= settings.TOP_K_RERANK:
                break

        logger.info(f"二次过滤完成: {len(reranked_results)} -> {len(filtered)} 条有效证据")
        return filtered

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        logger.info(f"开始检索: {query}")

        dense_results = self._dense_retrieve(query, settings.TOP_K_DENSE)
        sparse_results = self._sparse_retrieve(query, settings.TOP_K_SPARSE)

        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        candidates_for_rerank = fused_results[: settings.TOP_K_RERANK * 3]
        reranked_results = self._rerank(query, candidates_for_rerank)
        final_results = self._secondary_filter(reranked_results)

        logger.info(f"检索完成，返回 {len(final_results)} 条证据")
        return final_results
