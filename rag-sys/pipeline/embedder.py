import os
from threading import Lock
from typing import Optional

from sentence_transformers import CrossEncoder, SentenceTransformer

_MODEL: Optional[SentenceTransformer] = None
_MODEL_LOCK = Lock()

# Đường dẫn model reranker local (sau khi tải về, ví dụ: huggingface-cli download).
# Có thể override bằng biến môi trường LOCAL_RERANKER_PATH.
# Ví dụ đường dẫn tuyệt đối (Windows):
#   E:/programming/my_project/liga_rag_agent/sys/local_models/bge-reranker-base
_LOCAL_RERANKER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "local_models", "bge-reranker-base")
)
LOCAL_RERANKER_PATH = os.getenv("LOCAL_RERANKER_PATH", _LOCAL_RERANKER_DIR)

_RERANKER_MODEL: Optional[CrossEncoder] = None
_RERANKER_LOCK = Lock()


def get_embedding_model() -> SentenceTransformer:
    """
    Return a shared SentenceTransformer instance.
    Defaults to BAAI/bge-m3 if EMBEDDING_MODEL_NAME is not set.
    """
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
                _MODEL = SentenceTransformer(model_name)
    return _MODEL

def generate_embedding(text: str) -> list[float]:
    """Biến 1 đoạn text thành mảng 1024 con số"""
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Nhúng nhiều đoạn cùng lúc (cùng chuẩn normalize_embeddings=True như generate_embedding).
    Dùng cho ingest hàng loạt để tránh gọi encode từng chunk.
    """
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def warmup_embedding_model() -> None:
    """
    Load model into memory and run one tiny encode pass.
    Call this during app startup so requests are not delayed on first embed.
    """
    model = get_embedding_model()
    model.encode(["warmup"], show_progress_bar=False)


def warmup_reranker() -> None:
    """Load CrossEncoder và chạy một predict nhỏ để request đầu không bị trễ cold start."""
    reranker = get_reranker()
    reranker.predict([("warmup", "warmup passage")], show_progress_bar=False)


def get_reranker() -> CrossEncoder:
    """
    Singleton Cross-Encoder (BAAI/bge-reranker-base) cho bước rerank sau retrieval.
    CrossEncoder không dùng normalize_embeddings; chỉ cần predict scores cho cặp (query, doc).
    Nếu thư mục LOCAL_RERANKER_PATH chưa tồn tại, tải từ Hugging Face Hub.
    """
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        with _RERANKER_LOCK:
            if _RERANKER_MODEL is None:
                if os.path.isdir(LOCAL_RERANKER_PATH):
                    _RERANKER_MODEL = CrossEncoder(LOCAL_RERANKER_PATH)
                else:
                    _RERANKER_MODEL = CrossEncoder("BAAI/bge-reranker-base")
    return _RERANKER_MODEL
