from __future__ import annotations

import os
from threading import Lock

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

load_dotenv()

_LLM_LOCK = Lock()
_LLM: BaseChatModel | None = None


def get_deepseek_flash() -> BaseChatModel:
    """
    Một client ChatOpenAI dùng chung (singleton) để tránh tạo client/HTTP stack lặp lại mỗi node.
    max_tokens mặc định cho câu trả lời dài; node chỉ cần JSON ngắn dùng .bind(max_tokens=...).
    """
    global _LLM
    if _LLM is None:
        with _LLM_LOCK:
            if _LLM is None:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                base_url = os.getenv("DEEPSEEK_API_BASE")
                if not api_key:
                    raise ValueError("DEEPSEEK_API_KEY is not set")
                default_max = int(os.getenv("DEEPSEEK_MAX_TOKENS", "1000"))
                _LLM = ChatOpenAI(
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                    api_key=api_key,
                    base_url=base_url,
                    temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1")),
                    max_tokens=default_max,
                    request_timeout=float(os.getenv("DEEPSEEK_TIMEOUT_SEC", "120")),
                    max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "2")),
                )
    return _LLM
