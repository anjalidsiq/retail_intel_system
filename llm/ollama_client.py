#!/usr/bin/env python3
import logging
import os
from functools import lru_cache
from typing import List, Dict, Optional

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.output_parsers import StrOutputParser
    from langchain.prompts import ChatPromptTemplate


class OllamaClient:
    """LangChain wrapper for Ollama API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.default_model = default_model or os.getenv("OLLAMA_MODEL") or "llama2"
        self._parser = StrOutputParser()
        logging.info(f"[OllamaClient] Base URL: {self.base_url}, Model: {self.default_model}")

    @lru_cache(maxsize=8)
    def _base_llm(self, model: str, temperature: float):
        """Create and cache LLM instance."""
        return ChatOllama(base_url=self.base_url, model=model, temperature=temperature)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send messages to Ollama and get response."""
        prompt = ChatPromptTemplate.from_messages(
            [(msg.get("role", "user"), msg.get("content", "")) for msg in messages]
        )
        active_model = model or self.default_model
        llm = self._base_llm(active_model, temperature)
        
        chain = prompt | llm | self._parser
        return chain.invoke({})

    def healthcheck(self) -> bool:
        """Check if Ollama is responding."""
        try:
            response = self.chat(
                [{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=2
            )
            return "pong" in response.lower() or len(response.strip()) > 0
        except Exception as e:
            logging.error(f"[Ollama healthcheck] Failed: {e}")
            return False