# llm.py — shared LLM factory to avoid repeated ChatGroq instantiation
from langchain_groq import ChatGroq
from config import cfg


def get_llm(*, temperature: float = 0, streaming: bool = False) -> ChatGroq:
    """Return a configured ChatGroq instance.

    Centralises model selection so every call-site stays in sync
    when the model or defaults change in ``config.py``.
    """
    return ChatGroq(
        model=cfg.llm_model,
        temperature=temperature,
        streaming=streaming,
    )
