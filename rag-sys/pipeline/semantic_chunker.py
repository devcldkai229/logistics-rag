"""
Chunk Markdown (sau LlamaParse) bằng MarkdownTextSplitter — giữ cấu trúc heading/bảng tốt hơn split thuần ký tự.
"""

from langchain_text_splitters import MarkdownTextSplitter

# Kích thước tương thích RAG pháp lý (có thể chỉnh qua biến môi trường sau này).
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200


def split_markdown_to_chunks(markdown: str, *, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """
    Tách nội dung Markdown thành các đoạn văn bản (string) để lưu DB / embed.
    Trả về danh sách đã strip, bỏ chunk rỗng.
    """
    size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
    splitter = MarkdownTextSplitter(chunk_size=size, chunk_overlap=overlap)
    raw = splitter.split_text(markdown)
    return [c.strip() for c in raw if c and c.strip()]
