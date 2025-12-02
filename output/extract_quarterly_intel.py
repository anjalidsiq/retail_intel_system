"""Retail Intelligence Extraction Agent.

This module provides the extract_quarterly_intel function, which uses Llama 3.3 70B
to analyze a single earnings call transcript and extract structured intelligence
focused on Retail Media Investments, Supply Chain/Shelf Pain Points, and Category Strategy.
"""

import json
import logging
import re
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from llm.ollama_client import OllamaClient
from retail_ontology import (
    SYSTEM_PROMPT,
    JSON_SCHEMA_TEMPLATE,
    calculate_crm_segment,
    validate_extracted_data,
    generate_summary,
    calculate_confidence_score,
)
from ingestion.ingest_transcripts import EmbeddingProvider

def parse_intelligence_info(text_response: str) -> Dict[str, Any]:
    """Parse text response into structured intelligence data."""
    intelligence = {"retail_media_investments": "", "supply_chain_shelf_pain_points": "", "category_strategy": ""}
    
    lines = text_response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Retail Media:') or line.startswith('Media:'):
            intelligence["retail_media_investments"] = line.split(':', 1)[1].strip()
        elif line.startswith('Supply Chain:') or line.startswith('Shelf:'):
            intelligence["supply_chain_shelf_pain_points"] = line.split(':', 1)[1].strip()
        elif line.startswith('Strategy:') or line.startswith('Category:'):
            intelligence["category_strategy"] = line.split(':', 1)[1].strip()
    
    return {"intelligence": intelligence}

def parse_basic_info(text_response: str) -> Dict[str, Any]:
    """Parse text response into structured company and leadership data."""
    company_identity = {}
    leadership = []
    
    lines = text_response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Company:') or line.startswith('Brand:'):
            company_identity["brand_name"] = line.split(':', 1)[1].strip()
        elif line.startswith('Parent:'):
            company_identity["parent_company"] = line.split(':', 1)[1].strip()
        elif line.startswith('CEO:') or line.startswith('Executive:') or ':' in line:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    title_part = parts[0].strip()
                    name_part = parts[1].strip()
                    if 'CEO' in title_part or 'Chief' in title_part:
                        leadership.append({"name": name_part, "title": title_part})
    
    return {"company_identity": company_identity, "leadership": leadership}

def extract_quarterly_intel(transcript_text: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Extract structured intelligence from a single earnings call transcript.
    Uses comprehensive single-step extraction for improved accuracy.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Edge case: Empty or too short transcript
    if not transcript_text or len(transcript_text.strip()) < 100:
        raise ValueError("Transcript text is too short or empty. Minimum 100 characters required.")

    # Sanitize transcript: remove excessive whitespace
    transcript_text = re.sub(r'\s+', ' ', transcript_text.strip())

    client = OllamaClient(default_model="llama3.3:70b")
    
    # Single comprehensive extraction
    user_prompt = (
        "Extract intelligence from this earnings call transcript using the exact JSON schema from the system prompt.\n\n"
        "IMPORTANT:\n"
        "- Return ONLY valid JSON that matches the schema structure exactly\n"
        "- Fill all fields including nested objects in the intelligence section\n"
        "- Use empty strings for fields not mentioned in the text\n"
        "- leadership should be an array of objects with 'name' and 'title'\n"
        "- revenue_ttm should be a number\n\n"
        f"Transcript:\n{transcript_text[:15000]}"
    )
    
    try:
        response = client.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ], model="llama3.3:70b", temperature=0.1)
        
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        extracted_data = json.loads(response)
        
        # Normalize the data structure
        normalized_data = {
            "company_identity": extracted_data.get("company_identity", {}),
            "firmographics": extracted_data.get("firmographics", {"description": "", "industry_category": "", "revenue_ttm": 0, "currency": "USD", "employee_count": 0, "crm_segment": "SMB"}),
            "leadership": extracted_data.get("leadership", []),
            "intelligence": extracted_data.get("intelligence", {})
        }
        
        # If company info is in different fields, migrate it
        if "company" in extracted_data and not normalized_data["company_identity"].get("brand_name"):
            normalized_data["company_identity"]["brand_name"] = extracted_data["company"]
        
        if not validate_extracted_data(normalized_data):
            raise ValueError("Extracted data does not match required schema.")
        
        # Post-process: calculate crm_segment
        if "firmographics" in normalized_data and "revenue_ttm" in normalized_data["firmographics"]:
            revenue = normalized_data["firmographics"]["revenue_ttm"]
            normalized_data["firmographics"]["crm_segment"] = calculate_crm_segment(revenue)
        
        # Add metadata
        normalized_data["_metadata"] = {
            "extraction_timestamp": datetime.now().isoformat(),
            "model_used": "llama3.3:70b",
            "confidence_score": calculate_confidence_score(normalized_data),
            "summary": generate_summary(normalized_data),
            "extraction_method": "single_step_comprehensive"
        }
        
        logger.info("Successfully extracted intelligence for transcript using comprehensive method.")
        return normalized_data
    
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise

load_dotenv()

COLLECTION_NAME = "RetailTranscriptChunk"

def init_weaviate_client(logger: logging.Logger) -> weaviate.Client:
    """Initialize Weaviate client."""
    try:
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        auth = AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
        client = weaviate.Client(url=weaviate_url, auth_client_secret=auth)
        logger.info(f"Connected to Weaviate at {weaviate_url}")
        return client
    except Exception as e:
        logger.error(f"Error initializing Weaviate client: {e}")
        raise

def embed_query(query: str, embedder) -> List[float]:
    """Embed the query using the provided embedder."""
    try:
        return embedder.embed(query)
    except Exception as e:
        logging.error(f"Error embedding query: {e}")
        raise

def get_all_companies_from_weaviate(client: weaviate.Client) -> List[str]:
    """Dynamically retrieve all unique company names from Weaviate collection."""
    try:
        # Use a query to get all unique company values
        response = (
            client.query
            .get(COLLECTION_NAME, ["company"])
            .with_limit(1000)  # Get enough to cover all companies
            .do()
        )
        
        companies = set()
        for item in response["data"]["Get"][COLLECTION_NAME]:
            if item.get("company"):
                companies.add(item["company"])
        
        return list(companies)
    except Exception as e:
        logging.error(f"Error retrieving companies from Weaviate: {e}")
        return []

def extract_company_from_query(query: str, available_companies: List[str]) -> Optional[str]:
    """Extract company name from query using fuzzy matching against available companies."""
    from difflib import get_close_matches
    
    query_lower = query.lower()
    
    # First, try exact matches of company names in query
    for company in available_companies:
        if company.lower() in query_lower:
            return company
    
    # Extract potential company words from query (split by spaces and common separators)
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Look for fuzzy matches between query words and company names
    for company in available_companies:
        company_lower = company.lower()
        # Check if any significant part of the company name matches query words
        company_parts = set(re.findall(r'\b\w+\b', company_lower))
        
        # If company parts overlap with query words, it's a potential match
        if company_parts & query_words:
            return company
        
        # Also check if query words are substrings of company name (handles partial matches)
        for word in query_words:
            if len(word) > 3 and word in company_lower:
                return company
        
        # Try fuzzy matching on individual words
        for word in query_words:
            if len(word) > 3:  # Only check meaningful words
                matches = get_close_matches(word, [company_lower], n=1, cutoff=0.8)
                if matches:
                    return company
    
    return None

def query_weaviate(client: weaviate.Client, query_vector: List[float], query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Query Weaviate for similar chunks with optional company filtering."""
    # Get all available companies dynamically
    available_companies = get_all_companies_from_weaviate(client)
    
    # Try company-filtered search first
    company_filter = extract_company_from_query(query, available_companies)
    
    if company_filter:
        try:
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
            
            # If we got enough results from the filtered search, return them
            min_results = min(limit // 2, 3)  # At least half the limit or 3 results
            if len(filtered_results) >= min_results:
                return filtered_results
        except Exception as e:
            # If company filtering fails, fall back to general search
            pass
    
    # Fallback to general search if company filtering didn't yield enough results
    try:
        response = (
            client.query
            .get(COLLECTION_NAME, ["text", "company", "quarter", "page", "concept_hits"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .do()
        )
        return response["data"]["Get"][COLLECTION_NAME]
    except Exception as e:
        logging.error(f"Error querying Weaviate: {e}")
        raise

def aggregate_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Aggregate chunk texts into a single transcript-like string."""
    try:
        aggregated = []
        for chunk in chunks:
            aggregated.append(f"Company: {chunk.get('company', 'Unknown')}, Quarter: {chunk.get('quarter', 'Unknown')}, Page: {chunk.get('page', 'Unknown')}\n{chunk['text']}")
        return "\n\n".join(aggregated)
    except Exception as e:
        logging.error(f"Error aggregating chunks: {e}")
        raise

def query_intelligence(query: str, logger: Optional[logging.Logger] = None, limit: int = 20) -> Dict[str, Any]:
    """
    Query the intelligence database and extract structured data.

    Args:
        query: User query string.
        logger: Optional logger.
        limit: Number of top chunks to retrieve.

    Returns:
        Dict containing the extracted JSON data.

    Raises:
        ValueError: If query is invalid or no results.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not query or len(query.strip()) < 5:
        raise ValueError("Query too short. Minimum 5 characters required.")

    # Initialize clients
    client = init_weaviate_client(logger)
    ollama_client = OllamaClient(default_model="llama3.3:70b")
    
    # For embedding
    embedder = EmbeddingProvider(logger, True)

    # Embed query
    query_vector = embed_query(query, embedder)
    
    # Query Weaviate
    chunks = query_weaviate(client, query_vector, query, limit)
    if not chunks:
        raise ValueError("No relevant chunks found in the database.")
    
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query}")
    
    # Aggregate texts
    aggregated_text = aggregate_chunks(chunks)
    
    # Now, use the extraction logic on aggregated_text
    user_prompt = (
        "Extract intelligence from these aggregated transcript excerpts using the EXACT JSON schema structure.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- Return ONLY valid JSON that matches the schema structure exactly\n"
        "- Fill ALL fields in the intelligence section, including nested objects\n"
        "- Use EXACT field names: retail_media_budget, supply_chain_investment, digital_transformation_commitment, etc.\n"
        "- strategic_priorities must be an array of strings\n"
        "- Use empty strings \"\" for fields not mentioned in the text\n"
        "- Do not create new fields or modify the schema structure\n"
        "- leadership should be an array of objects with 'name' and 'title' fields\n"
        "- revenue_ttm should be a number (0 if not mentioned)\n\n"
        f"Excerpts:\n{aggregated_text[:15000]}"
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Use production model for full capability
        response = ollama_client.chat(messages, model="llama3.3:70b", temperature=0.1)
        response = response.strip()
        
        # Debug: Log the raw LLM response before stripping
        logger.info(f"Raw LLM response before stripping: {response[:2000]}...")
        
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        # Extract JSON from response if it contains explanatory text
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        
        # Debug: Log the processed response
        logger.info(f"Processed LLM response: {response[:2000]}...")
        
        extracted_data = json.loads(response)
        
        # Debug: Log the raw LLM response
        logger.info(f"Raw LLM response: {json.dumps(extracted_data, indent=2)[:1000]}...")
        
        # Use comprehensive normalization to map LLM creative fields to exact schema
        from retail_ontology import comprehensive_normalize_data
        normalized_data = comprehensive_normalize_data(extracted_data)
        
        # Debug: Log the normalized data
        logger.info(f"Normalized data: {json.dumps(normalized_data, indent=2)[:1000]}...")
        
        if not validate_extracted_data(normalized_data):
            logger.error("Schema validation failed after normalization. Normalized data structure:")
            logger.error(json.dumps(normalized_data, indent=2))
            raise ValueError("Extracted data does not match required schema.")
        
        if "firmographics" in extracted_data and "revenue_ttm" in extracted_data["firmographics"]:
            revenue = extracted_data["firmographics"]["revenue_ttm"]
            extracted_data["firmographics"]["crm_segment"] = calculate_crm_segment(revenue)
        
        # Add metadata with source information
        normalized_data["_metadata"] = {
            "extraction_timestamp": datetime.now().isoformat(),
            "model_used": "llama3.3:70b",
            "confidence_score": calculate_confidence_score(normalized_data, len(chunks)),
            "summary": generate_summary(normalized_data),
            "source_chunks_used": len(chunks),
            "query_used": query
        }
        
        logger.info("Successfully queried and extracted intelligence.")
        return normalized_data
    
    except json.JSONDecodeError as e:
        logger.error("LLM response is not valid JSON: %s", e)
        raise ValueError("Invalid JSON response from LLM") from e
    except Exception as e:
        logger.error("Error during query extraction: %s", e)
        raise

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retail Intelligence Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract intelligence from a transcript")
    extract_parser.add_argument("--transcript-file", type=str, help="Path to transcript file")
    extract_parser.add_argument("--transcript-text", type=str, help="Transcript text directly")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query intelligence from database")
    query_parser.add_argument("query", type=str, help="Query string")
    query_parser.add_argument("--limit", type=int, default=20, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.command == "extract":
        if args.transcript_file:
            # Check if it's a PDF file
            if args.transcript_file.lower().endswith('.pdf'):
                from pypdf import PdfReader
                reader = PdfReader(args.transcript_file)
                transcript_pages = []
                for page in reader.pages:
                    text = page.extract_text() or ""
                    transcript_pages.append(text)
                transcript = "\n\n".join(transcript_pages)
            else:
                with open(args.transcript_file, 'r') as f:
                    transcript = f.read()
        elif args.transcript_text:
            transcript = args.transcript_text
        else:
            parser.error("Must provide --transcript-file or --transcript-text for extract")
        
        try:
            result = extract_quarterly_intel(transcript, logger)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.command == "query":
        try:
            result = query_intelligence(args.query, logger, args.limit)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        parser.print_help()