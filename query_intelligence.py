"""Query Intelligence Pipeline.

This module provides the query_intelligence function, which takes a user query,
searches Weaviate for relevant transcript chunks, aggregates them, and uses Llama 3.3 70B
to extract structured intelligence.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from llm.ollama_client import OllamaClient
from retail_ontology import SYSTEM_PROMPT, JSON_SCHEMA_TEMPLATE, calculate_crm_segment, validate_extracted_data, COLLECTION_NAME
from ingestion.ingest_transcripts import EmbeddingProvider

# Load environment variables
load_dotenv()

def init_weaviate_client(logger: logging.Logger) -> weaviate.Client:
    """Initialize Weaviate client."""
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    auth = AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
    client = weaviate.Client(url=weaviate_url, auth_client_secret=auth)
    logger.info(f"Connected to Weaviate at {weaviate_url}")
    return client

def embed_query(query: str, embedder) -> List[float]:
    """Embed the query using the provided embedder."""
    return embedder.embed(query)

def extract_company_from_query(query: str) -> Optional[str]:
    """Extract company name from query for filtering."""
    # Common company name mappings
    company_mappings = {
        'mondelez': 'Mondelezinternational',
        'mondelez international': 'Mondelezinternational',
        'unilever': 'Unilever'
    }
    
    query_lower = query.lower()
    for key, value in company_mappings.items():
        if key in query_lower:
            return value
    
    return None

def query_weaviate(client: weaviate.Client, query_vector: List[float], query: str, limit: int = 10, logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """Query Weaviate for similar chunks with optional company filtering."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Try company-filtered search first
    company_filter = extract_company_from_query(query)
    
    if company_filter:
        logger.info(f"Attempting company-filtered search for: {company_filter}")
        # First, try to get results filtered by company
        response = (
            client.query
            .get(COLLECTION_NAME, ["text", "company", "quarter", "page", "concept_hits"])
            .with_near_vector({"vector": query_vector})
            .with_where({
                "path": ["company"],
                "operator": "Equal",
                "valueString": company_filter
            })
            .with_limit(limit)
            .do()
        )
        
        filtered_results = response["data"]["Get"][COLLECTION_NAME]
        logger.info(f"Company-filtered search returned {len(filtered_results)} results")
        
        # If we got enough results from the filtered search, return them
        min_results = min(limit // 2, 3)  # At least half the limit or 3 results
        if len(filtered_results) >= min_results:
            logger.info(f"Using company-filtered results ({len(filtered_results)} chunks)")
            return filtered_results
        else:
            logger.info(f"Company-filtered search returned only {len(filtered_results)} results, falling back to general search")
    
    # Fallback to general search if company filtering didn't yield enough results
    logger.info("Performing general vector search")
    response = (
        client.query
        .get(COLLECTION_NAME, ["text", "company", "quarter", "page", "concept_hits"])
        .with_near_vector({"vector": query_vector})
        .with_limit(limit)
        .do()
    )
    
    general_results = response["data"]["Get"][COLLECTION_NAME]
    logger.info(f"General search returned {len(general_results)} results")
    return general_results

def aggregate_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Aggregate chunk texts into a single transcript-like string."""
    aggregated = []
    for chunk in chunks:
        aggregated.append(f"Company: {chunk.get('company', 'Unknown')}, Quarter: {chunk.get('quarter', 'Unknown')}, Page: {chunk.get('page', 'Unknown')}\n{chunk['text']}")
    return "\n\n".join(aggregated)

def query_intelligence(query: str, logger: Optional[logging.Logger] = None, limit: int = 20) -> Dict[str, Any]:
    """
    Query the intelligence database and extract structured data.

    Args:
        query: User query string.
        logger: Optional logger.
        limit: Number of top chunks to retrieve.

    Returns:
        Dict containing the extracted JSON data.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not query or len(query.strip()) < 5:
        raise ValueError("Query too short. Minimum 5 characters required.")

    # Initialize clients
    client = init_weaviate_client(logger)
    ollama_client = OllamaClient(default_model="llama3.3:70b")
    
    # For embedding, we need the embedder. Assuming nomic-embed-text
    embedder = EmbeddingProvider(logger, True)

    # Embed query
    query_vector = embed_query(query, embedder)
    
    # Query Weaviate
    chunks = query_weaviate(client, query_vector, query, limit, logger)
    if not chunks:
        raise ValueError("No relevant chunks found in the database.")
    
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query}")
    for i, chunk in enumerate(chunks[:3]):  # Log first 3
        logger.info(f"Chunk {i}: {chunk['text'][:200]}...")
    
    # Aggregate texts
    aggregated_text = aggregate_chunks(chunks)
    logger.info(f"Aggregated text length: {len(aggregated_text)}, first 500 chars: {aggregated_text[:500]}")
    
    # Now, use the extraction logic on aggregated_text
    user_prompt = (
        "Analyze the following aggregated transcript excerpts and fill the JSON schema with extracted data.\n\n"
        f"Schema Template:\n{json.dumps(JSON_SCHEMA_TEMPLATE, indent=2).replace('{', '{{').replace('}', '}}')}\n\n"
        f"Excerpts:\n{aggregated_text[:15000]}"
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = ollama_client.chat(messages, model="llama3.3:70b", temperature=0.1)
        logger.info(f"LLM raw response (first 500 chars): {response[:500]}")
        response = response.strip()
        
        # Extract JSON from markdown code blocks or plain JSON
        if '```json' in response:
            start_idx = response.find('```json') + 7
            end_idx = response.find('```', start_idx)
            response = response[start_idx:end_idx]
        elif '```' in response:
            start_idx = response.find('```') + 3
            end_idx = response.find('```', start_idx)
            response = response[start_idx:end_idx]
        
        response = response.strip()
        logger.info(f"LLM processed response (first 500 chars): {response[:500]}")
        extracted_data = json.loads(response)
        
        if not validate_extracted_data(extracted_data):
            raise ValueError("Extracted data does not match required schema.")
        
        if "firmographics" in extracted_data and "revenue_ttm" in extracted_data["firmographics"]:
            revenue = extracted_data["firmographics"]["revenue_ttm"]
            extracted_data["firmographics"]["crm_segment"] = calculate_crm_segment(revenue)
        
        logger.info("Successfully queried and extracted intelligence.")
        return extracted_data
    
    except json.JSONDecodeError as e:
        logger.error("LLM response is not valid JSON: %s", e)
        logger.error(f"Failed response was: {response[:1000]}")
        raise ValueError("Invalid JSON response from LLM") from e
    except Exception as e:
        logger.error("Error during query extraction: %s", e)
        raise

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        result = query_intelligence("[Company Name] Q3 2025 retail media supply chain strategy performance leadership", limit=10)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")