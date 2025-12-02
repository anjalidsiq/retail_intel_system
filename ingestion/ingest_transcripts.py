"""Retail transcript ingestion utility for Weaviate.

A single-file script that:
- Loads environment variables and CLI args
- Ensures the `RetailTranscriptChunk` collection exists with ideal schema
- Reads PDFs, chunks text, and tags concept hits from retail ontology
- Optionally deletes existing chunks before re‑ingesting (CRUD support)
- Generates embeddings via Ollama (if configured) or lets Weaviate vectorize
- Handles errors gracefully and logs actionable information
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
import requests
from pypdf import PdfReader
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.exceptions import WeaviateBaseError
from retail_ontology import RETAIL_CONCEPTS

COLLECTION_NAME = "RetailTranscriptChunk"
MIN_CHUNK_CHARS = 240
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP = 200
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

QUARTER_PATTERN = re.compile(r"(Q[1-4])", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(20\d{2})")
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class ScriptConfig:
    input_dir: Path
    delete_existing: bool
    source_url: Optional[str]
    call_type: Optional[str]
    language: str
    chunk_chars: int
    chunk_overlap: int
    use_local_embeddings: bool
    env_file: Optional[Path]
    verbose: bool


class EmbeddingProvider:
    """Optional embedding provider using Ollama via LangChain."""

    def __init__(self, logger: logging.Logger, enabled: bool) -> None:
        self.logger = logger
        self.enabled = enabled
        self._model = None
        self._embedder = None
        self._base_url: Optional[str] = None
        if enabled:
            self._setup()

    def _setup(self) -> None:
        try:
            from langchain_ollama import OllamaEmbeddings

            model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            base_url = os.getenv("OLLAMA_BASE_URL")
            if not base_url:
                raise ValueError("OLLAMA_BASE_URL is not set")
            self._base_url = base_url.rstrip("/")
            self._assert_health()
            self._embedder = OllamaEmbeddings(model=model, base_url=self._base_url)
            self._model = model
            self.logger.info("Using Ollama embeddings model %s", model)
        except Exception as exc:  # pragma: no cover - fallback
            self.logger.warning(
                "Falling back to Weaviate vectorizer (Ollama embeddings unavailable): %s",
                exc,
            )
            self.enabled = False

    def _assert_health(self) -> None:
        if not self._base_url:
            return
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - validation only
            raise RuntimeError(f"Ollama healthcheck failed: {exc}") from exc

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.enabled or not self._embedder:
            return None
        if not text.strip():
            return None
        try:
            return self._embedder.embed_query(text)
        except Exception as exc:  # pragma: no cover - best-effort fallback
            self.logger.warning(
                "Embedding call failed, falling back to server-side vectorization: %s",
                exc,
            )
            return None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def run_with_retries(
    operation_name: str,
    logger: logging.Logger,
    func,
    *,
    retries: int = MAX_RETRIES,
    backoff_seconds: int = RETRY_BACKOFF_SECONDS,
):
    for attempt in range(1, retries + 1):
        try:
            return func()
        except WeaviateBaseError as exc:
            if attempt == retries:
                logger.error(
                    "%s failed after %s attempts: %s", operation_name, retries, exc
                )
                raise
            sleep_time = backoff_seconds * attempt
            logger.warning(
                "%s attempt %s/%s failed: %s. Retrying in %ss",
                operation_name,
                attempt,
                retries,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)


def setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)5s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("retail_ingest")
    logger.setLevel(level)
    return logger


def load_environment(env_file: Optional[Path]) -> None:
    if env_file:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()


def resolve_vectorizer_setting() -> str:
    value = os.getenv("WEAVIATE_VECTORIZER", "text2vec-transformers").strip()
    return value or "text2vec-transformers"


def init_weaviate_client(logger: logging.Logger) -> weaviate.Client:
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.getenv("WEAVIATE_API_KEY")
    timeout = (5, 60)
    auth = AuthApiKey(api_key) if api_key else None
    logger.info("Connecting to Weaviate at %s", url)
    return weaviate.Client(url, auth_client_secret=auth, timeout_config=timeout)


def ensure_collection(
    client: weaviate.Client,
    logger: logging.Logger,
    *,
    vectorizer: str,
) -> None:
    try:
        classes = client.schema.get().get("classes", [])
    except WeaviateBaseError as exc:
        logger.error("Unable to read schema: %s", exc)
        raise

    if any(cls.get("class") == COLLECTION_NAME for cls in classes):
        logger.debug("Collection %s already exists", COLLECTION_NAME)
        return

    schema = {
        "class": COLLECTION_NAME,
        "description": "Retail earnings transcript chunks tagged with domain concepts",
        "vectorizer": vectorizer,
        "properties": [
            {"name": "company", "dataType": ["text"], "description": "Brand name"},
            {"name": "parent_company", "dataType": ["text"]},
            {"name": "ticker", "dataType": ["text"]},
            {"name": "exchange", "dataType": ["text"]},
            {"name": "quarter", "dataType": ["text"]},
            {"name": "year", "dataType": ["int"]},
            {"name": "call_type", "dataType": ["text"]},
            {"name": "language", "dataType": ["text"]},
            {"name": "source_url", "dataType": ["text"]},
            {"name": "page", "dataType": ["int"]},
            {"name": "chunk_index", "dataType": ["int"]},
            {"name": "text", "dataType": ["text"]},
            {"name": "concept_hits", "dataType": ["text[]"]},
            {"name": "concept_score", "dataType": ["number"]},
            {"name": "created_at", "dataType": ["date"]},
            {"name": "updated_at", "dataType": ["date"]},
        ],
    }

    if vectorizer == "text2vec-transformers":
        schema["moduleConfig"] = {"text2vec-transformers": {"vectorizeClassName": False}}
    elif vectorizer == "text2vec-openai":
        schema["moduleConfig"] = {"text2vec-openai": {"vectorizeClassName": False}}
    elif vectorizer == "none":
        schema["moduleConfig"] = {}
    else:
        schema["moduleConfig"] = {}

    client.schema.create_class(schema)
    logger.info("Created collection %s", COLLECTION_NAME)


def delete_existing_chunks(
    client: weaviate.Client,
    logger: logging.Logger,
    *,
    company: Optional[str] = None,
    ticker: Optional[str] = None,
    quarter: Optional[str] = None,
    year: Optional[int] = None,
) -> None:
    filters = []
    if company:
        filters.append({"path": ["company"], "operator": "Equal", "valueText": company})
    if ticker:
        filters.append({"path": ["ticker"], "operator": "Equal", "valueText": ticker})
    if quarter:
        filters.append({"path": ["quarter"], "operator": "Equal", "valueText": quarter})
    if year:
        filters.append({"path": ["year"], "operator": "Equal", "valueInt": year})

    if not filters:
        logger.info("No metadata filters available; skipping deleteExisting step")
        return

    # Combine filters with AND
    where = filters[0]
    for clause in filters[1:]:
        where = {"operator": "And", "operands": [where, clause]}

    def _delete():
        return client.batch.delete_objects(
            class_name=COLLECTION_NAME,
            where=where,
            output="minimal",
        )

    result = run_with_retries("delete_objects", logger, _delete)
    logger.info("Deleted existing chunks: %s", result)


# ---------------------------------------------------------------------------
# PDF + chunk processing
# ---------------------------------------------------------------------------

def extract_pdf_pages(pdf_path: Path, logger: logging.Logger) -> List[Tuple[int, str]]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pragma: no cover
        logger.error("Failed reading %s: %s", pdf_path.name, exc)
        return []

    pages: List[Tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover
            logger.warning("Unable to extract text from %s page %s: %s", pdf_path.name, idx, exc)
            text = ""
        pages.append((idx, text))
    return pages


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = max(end - overlap, 0)
        if start == 0 and len(chunks) > 1:
            break  # avoid infinite loop on short text
    return chunks


def normalize_chunk_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_meaningful_chunk(text: str) -> bool:
    if not text:
        return False
    alpha_count = sum(1 for ch in text if ch.isalpha())
    ratio = alpha_count / max(len(text), 1)
    return ratio >= 0.15


def chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def tag_concepts(chunk: str) -> Tuple[List[str], float]:
    lowered = chunk.lower()
    hits: List[str] = []
    for category, terms in RETAIL_CONCEPTS.items():
        for term in terms:
            if term.lower() in lowered:
                hits.append(f"{category}:{term}")
    return hits, float(len(hits))


def parse_filename_metadata(pdf_path: Path) -> Dict[str, Optional[str]]:
    stem = pdf_path.stem
    normalized = re.sub(r"[\s\-]+", "_", stem)
    tokens = [token for token in normalized.split("_") if token]

    metadata: Dict[str, Optional[str]] = {
        "company": None,
        "ticker": None,
        "quarter": None,
        "year": None,
    }

    quarter_match = QUARTER_PATTERN.search(stem)
    if quarter_match:
        metadata["quarter"] = quarter_match.group(1).upper()

    year_match = YEAR_PATTERN.search(stem)
    if year_match:
        metadata["year"] = int(year_match.group(1))

    company_tokens: List[str] = []
    for token in tokens:
        upper = token.upper()
        if metadata["quarter"] and upper == metadata["quarter"]:
            continue
        if metadata["year"] and token == str(metadata["year"]):
            continue
        if not metadata["ticker"] and TICKER_PATTERN.fullmatch(upper):
            metadata["ticker"] = upper
            continue
        company_tokens.append(token)

    if company_tokens:
        raw_company = " ".join(company_tokens)
        metadata["company"] = raw_company.replace("_", " ").title()
    else:
        metadata["company"] = stem.replace("_", " ")

    return metadata


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def prepare_object(
    *,
    chunk: str,
    chunk_index: int,
    page: int,
    metadata: Dict[str, Optional[str]],
    source_url: Optional[str],
    call_type: Optional[str],
    language: str,
) -> Dict:
    hits, score = tag_concepts(chunk)
    now = datetime.now(timezone.utc).isoformat()
    return {
        "company": metadata.get("company") or "",
        "parent_company": metadata.get("parent_company") or "",
        "ticker": metadata.get("ticker") or "",
        "exchange": metadata.get("exchange") or "",
        "quarter": metadata.get("quarter") or "",
        "year": metadata.get("year") or 0,
        "call_type": call_type or "",
        "language": language,
        "source_url": source_url or "",
        "page": page,
        "chunk_index": chunk_index,
        "text": chunk,
        "concept_hits": hits,
        "concept_score": score,
        "created_at": now,
        "updated_at": now,
    }


def upsert_chunk(
    client: weaviate.Client,
    logger: logging.Logger,
    obj: Dict,
    *,
    vector: Optional[List[float]],
) -> None:
    obj_id = str(uuid.uuid4())
    def _create():
        client.data_object.create(
            data_object=obj,
            class_name=COLLECTION_NAME,
            uuid=obj_id,
            vector=vector,
        )

    run_with_retries("data_object.create", logger, _create)


def ingest_pdf(
    pdf_path: Path,
    *,
    config: ScriptConfig,
    client: weaviate.Client,
    logger: logging.Logger,
    embedder: EmbeddingProvider,
    vector_required: bool,
) -> None:
    metadata = parse_filename_metadata(pdf_path)
    logger.info(
        "Processing %s (company=%s ticker=%s quarter=%s year=%s)",
        pdf_path.name,
        metadata.get("company"),
        metadata.get("ticker"),
        metadata.get("quarter"),
        metadata.get("year"),
    )

    if config.delete_existing and any(
        metadata.get(field)
        for field in ("company", "ticker", "quarter", "year")
    ):
        delete_existing_chunks(
            client,
            logger,
            company=metadata.get("company"),
            ticker=metadata.get("ticker"),
            quarter=metadata.get("quarter"),
            year=metadata.get("year"),
        )

    pages = extract_pdf_pages(pdf_path, logger)
    total_chunks = 0
    seen_chunk_hashes: set[str] = set()

    for page_number, text in pages:
        page_chunks = chunk_text(text, config.chunk_chars, config.chunk_overlap)
        for idx, chunk in enumerate(page_chunks):
            normalized_chunk = normalize_chunk_text(chunk)
            if len(normalized_chunk) < MIN_CHUNK_CHARS:
                continue
            if not is_meaningful_chunk(normalized_chunk):
                continue
            signature = chunk_hash(normalized_chunk)
            if signature in seen_chunk_hashes:
                continue
            seen_chunk_hashes.add(signature)
            obj = prepare_object(
                chunk=normalized_chunk,
                chunk_index=total_chunks,
                page=page_number,
                metadata=metadata,
                source_url=config.source_url,
                call_type=config.call_type,
                language=config.language,
            )
            vector = embedder.embed(normalized_chunk)
            if vector_required and vector is None:
                logger.error(
                    "Vectorizer is 'none' but embeddings failed for chunk %s on page %s",
                    total_chunks,
                    page_number,
                )
                raise RuntimeError(
                    "Embeddings are required (vectorizer='none') but were not produced."
                )
            upsert_chunk(client, logger, obj, vector=vector)
            total_chunks += 1

    logger.info("Inserted %s chunks from %s", total_chunks, pdf_path.name)


def run_ingestion(config: ScriptConfig) -> None:
    logger = setup_logger(config.verbose)
    load_environment(config.env_file)
    client = init_weaviate_client(logger)
    vectorizer_setting = resolve_vectorizer_setting().lower()
    ensure_collection(client, logger, vectorizer=vectorizer_setting)
    need_local_embeddings = config.use_local_embeddings or vectorizer_setting == "none"
    embedder = EmbeddingProvider(logger, need_local_embeddings)

    if vectorizer_setting == "none" and not embedder.enabled:
        logger.error(
            "WEAVIATE_VECTORIZER is 'none' but embeddings are not configured. "
            "Set OLLAMA_BASE_URL/OLLAMA_EMBED_MODEL and rerun with --use-local-embeddings."
        )
        raise SystemExit(1)

    pdf_dir = config.input_dir
    if not pdf_dir.exists():
        logger.error("Input directory %s does not exist", pdf_dir)
        raise SystemExit(1)

    pdf_files = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return

    for pdf in pdf_files:
        ingest_pdf(
            pdf,
            config=config,
            client=client,
            logger=logger,
            embedder=embedder,
            vector_required=vectorizer_setting == "none",
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> ScriptConfig:
    parser = argparse.ArgumentParser(
        description="Ingest retail earnings transcripts into Weaviate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="data/transcripts", help="Folder containing PDFs")
    parser.add_argument("--source-url", default=None, help="Default source URL to attach")
    parser.add_argument("--call-type", default=None, help="Call type label (e.g., earnings, investor day)")
    parser.add_argument("--language", default="en", help="Language code to store on chunks")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_MAX_CHARS, help="Max characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_OVERLAP, help="Character overlap between chunks")
    parser.add_argument("--delete-existing", action="store_true", help="Remove existing chunks for the same transcript metadata before ingesting")
    parser.add_argument("--env-file", default=None, help="Path to .env file (defaults to project .env)")
    parser.add_argument("--use-local-embeddings", action="store_true", help="Use Ollama embeddings locally instead of Weaviate vectorizer")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    return ScriptConfig(
        input_dir=Path(args.input_dir),
        delete_existing=args.delete_existing,
        source_url=args.source_url,
        call_type=args.call_type,
        language=args.language,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        use_local_embeddings=args.use_local_embeddings,
        env_file=Path(args.env_file) if args.env_file else None,
        verbose=args.verbose,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    run_ingestion(config)


if __name__ == "__main__":
    main()
