#!/usr/bin/env python3
"""
Simplified intelligence extraction focusing on user schema output.

Takes chunks from Weaviate and produces:
- contact_data_map: array of leaders
- account_data_map: strategic summary, flags, focussed_category
- focussed_category_options: list of 15 categories
"""

import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey

from ingestion.ingest_transcripts import EmbeddingProvider
from llm.ollama_client import OllamaClient

load_dotenv()

# Category options
FOCUSSED_CATEGORY_OPTIONS = [
    "Beverages",
    "Snacks & Confectionery",
    "Pantry Staples",
    "Dairy & Chilled",
    "Frozen Foods",
    "Hair Care",
    "Skin Care",
    "Cosmetics",
    "Personal Hygiene",
    "Cleaning Supplies",
    "Paper Goods",
    "OTC Medication",
    "Vitamins & Supplements",
    "Baby Care",
    "Pet Food",
]

COLLECTION_NAME = "RetailTranscriptChunk"


def init_weaviate_client(logger: logging.Logger) -> weaviate.Client:
    """Initialize Weaviate client."""
    try:
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        auth = AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
        client = weaviate.Client(url=weaviate_url, auth_client_secret=auth)
        logger.info(f"✓ Connected to Weaviate at {weaviate_url}")
        return client
    except Exception as e:
        logger.error(f"✗ Error initializing Weaviate client: {e}")
        raise


def embed_query(query: str, embedder) -> List[float]:
    """Embed the query using the provided embedder."""
    try:
        return embedder.embed(query)
    except Exception as e:
        logging.error(f"Error embedding query: {e}")
        raise


def query_weaviate(client: weaviate.Client, query_vector: List[float], limit: int = 20) -> List[Dict[str, Any]]:
    """Query Weaviate for similar chunks."""
    try:
        response = (
            client.query
            .get(COLLECTION_NAME, ["text", "company", "quarter", "page", "concept_hits"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .do()
        )
        chunks = response["data"]["Get"][COLLECTION_NAME]
        logging.info(f"✓ Retrieved {len(chunks)} chunks from Weaviate")
        return chunks
    except Exception as e:
        logging.error(f"Error querying Weaviate: {e}")
        raise


def aggregate_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Aggregate chunk texts into a single transcript-like string."""
    try:
        aggregated = []
        for chunk in chunks:
            text = chunk.get('text', '')
            company = chunk.get('company', 'Unknown')
            quarter = chunk.get('quarter', 'Unknown')
            aggregated.append(f"[{company} {quarter}]\n{text}")
        return "\n\n---\n\n".join(aggregated)
    except Exception as e:
        logging.error(f"Error aggregating chunks: {e}")
        raise


def extract_with_llm(aggregated_text: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Call LLM to extract intelligence from aggregated chunks.
    
    Returns a simplified structure focused on:
    - strategic objectives and priorities
    - financial performance
    - risk factors
    - key initiatives
    """
    try:
        client = OllamaClient(default_model="llama3.3:70b")
        
        prompt = f"""Extract key intelligence from this earnings call or financial data.

IMPORTANT: Return ONLY valid JSON with this exact structure:
{{{{
  "strategic_priorities": ["priority 1", "priority 2"],
  "financial_summary": "brief summary of financial performance",
  "key_initiatives": ["initiative 1", "initiative 2"],
  "risks": ["risk 1", "risk 2"],
  "category_focus": "if mentioned, which category (e.g., Hair Care, Frozen Foods)",
  "retail_media_mentioned": true,
  "supply_chain_mentioned": false
}}}}

TEXT:
{aggregated_text[:10000]}
"""
        
        response = client.chat([
            {"role": "user", "content": prompt}
        ], model="llama3.3:70b", temperature=0.1)
        
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        extracted = json.loads(response)
        logger.info("✓ LLM extraction complete")
        return extracted
        
    except Exception as e:
        logger.error(f"Error during LLM extraction: {e}")
        raise


def build_user_schema(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    """Build the final user schema from LLM output."""
    
    # Extract strategic priority summary
    priorities = llm_output.get("strategic_priorities", [])
    financial = llm_output.get("financial_summary", "")
    initiatives = llm_output.get("key_initiatives", [])
    
    strategic_priority_summary = ""
    if financial:
        strategic_priority_summary = financial
    if priorities:
        priority_text = ", ".join(priorities)
        strategic_priority_summary += f"\n\n🎯 Priorities: {priority_text}"
    if initiatives:
        init_text = ", ".join(initiatives)
        strategic_priority_summary += f"\n\n📌 Initiatives: {init_text}"
    
    # Extract risk summary
    risks = llm_output.get("risks", [])
    risk_summary = ", ".join(risks) if risks else ""
    
    # Extract flags
    retail_media_flag = llm_output.get("retail_media_mentioned", False)
    supply_chain_flag = llm_output.get("supply_chain_mentioned", False)
    category_focus = llm_output.get("category_focus", "")
    
    # Determine focussed_category
    focussed_category = ""
    if category_focus:
        # Try exact match
        for opt in FOCUSSED_CATEGORY_OPTIONS:
            if opt.lower() in category_focus.lower():
                focussed_category = opt
                break
    
    category_flag = bool(category_focus) or bool(focussed_category)
    
    return {
        "contact_data_map": [],  # No leadership extraction in this simplified version
        "account_data_map": {
            "strategic_priority_summary": strategic_priority_summary,
            "risk_summary": risk_summary,
            "retail_media_intent_flag": bool(retail_media_flag),
            "supply_chain_issues_flag": bool(supply_chain_flag),
            "category_focus_flag": bool(category_flag),
            "focussed_category": focussed_category,
        }
    }


def query_and_extract(query: str, logger: Optional[logging.Logger] = None, limit: int = 20) -> Dict[str, Any]:
    """
    Main function: Query Weaviate and extract intelligence with simplified user schema output.
    
    Args:
        query: Search query (e.g., "Unilever Q1 2025")
        logger: Optional logger
        limit: Number of chunks to retrieve
    
    Returns:
        Dict with contact_data_map, account_data_map, focussed_category_options
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("SIMPLIFIED INTELLIGENCE EXTRACTION")
        logger.info("=" * 80)
        
        # Step 1: Initialize clients
        weaviate_client = init_weaviate_client(logger)
        embedder = EmbeddingProvider(logger, True)
        
        # Step 2: Embed query
        logger.info(f"Query: {query}")
        query_vector = embed_query(query, embedder)
        logger.info("✓ Query embedded")
        
        # Step 3: Query Weaviate
        chunks = query_weaviate(weaviate_client, query_vector, limit)
        
        if not chunks:
            logger.warning("No chunks found in Weaviate")
            return {
                "contact_data_map": [],
                "account_data_map": {
                    "strategic_priority_summary": "No data found for query: " + query,
                    "risk_summary": "",
                    "retail_media_intent_flag": False,
                    "supply_chain_issues_flag": False,
                    "category_focus_flag": False,
                    "focussed_category": "",
                }
            }
        
        # Step 4: Aggregate chunks
        aggregated_text = aggregate_chunks(chunks)
        logger.info(f"✓ Aggregated {len(chunks)} chunks")
        
        # Step 5: Call LLM
        llm_output = extract_with_llm(aggregated_text, logger)
        logger.info("✓ LLM processing complete")
        
        # Step 6: Build user schema
        result = build_user_schema(llm_output)
        logger.info("✓ Schema mapping complete")
        
        logger.info("=" * 80)
        logger.info("✓ EXTRACTION SUCCESSFUL")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        raise


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Retail Intelligence Extraction")
    parser.add_argument("--company", type=str, help="Company name")
    parser.add_argument("--quarter", type=str, help="Quarter (e.g., Q1)")
    parser.add_argument("--year", type=str, help="Year (e.g., 2025)")
    parser.add_argument("--query", type=str, help="Free-form query")
    parser.add_argument("--limit", type=int, default=20, help="Number of chunks")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Build query
    if args.company and args.quarter and args.year:
        query = f"{args.company} {args.quarter} {args.year}"
    elif args.query:
        query = args.query
    else:
        parser.error("Must provide either (--company --quarter --year) or (--query)")
    
    try:
        result = query_and_extract(query, logger, args.limit)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
