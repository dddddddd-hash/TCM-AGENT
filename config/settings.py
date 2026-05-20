from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com"
    LLM_MODEL_NAME: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.1

    EMBEDDING_MODEL_PATH: str = "BAAI/bge-large-zh-v1.5"
    EMBEDDING_DIMENSION: int = 1024

    RERANK_MODEL_PATH: str = "BAAI/bge-reranker-large"

    NER_MODEL_PATH: str = "your/tcm-ner-model"
    NLI_MODEL_PATH: str = "cross-encoder/nli-deberta-v3-base"

    FAISS_INDEX_PATH: str = "data/memory_store/faiss_hnsw.index"
    HNSW_M: int = 32
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 128

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.85

    TOP_K_DENSE: int = 20
    TOP_K_SPARSE: int = 20
    TOP_K_RERANK: int = 5
    RERANK_SCORE_THRESHOLD: float = 0.4

    SHORT_TERM_WINDOW_SIZE: int = 6
    SUMMARY_TRIGGER_TURNS: int = 4
    IMPORTANCE_DECAY_FACTOR: float = 0.85
    NER_WEIGHT_MAP: dict = {
        "疾病": 1.5,
        "症状": 1.3,
        "中药": 1.4,
        "方剂": 1.4,
        "穴位": 1.2,
        "治法": 1.3,
        "DEFAULT": 1.0,
    }
    LONG_TERM_IMPORTANCE_THRESHOLD: float = 0.6
    LONG_TERM_SIMILARITY_THRESHOLD: float = 0.75
    NLI_CONTRADICTION_THRESHOLD: float = 0.85

    MAX_REACT_ITERATIONS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.75
    EVIDENCE_CONSISTENCY_THRESHOLD: float = 0.6

    class Config:
        env_file = ".env"


settings = Settings()
