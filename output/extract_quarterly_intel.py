#!/usr/bin/env python3
"""Retail Intelligence Extraction Tool - Refactored.

This module extracts structured retail intelligence from earnings call transcripts
stored in Weaviate database. It accepts CLI queries with company name, quarter, and year
and produces structured JSON output with contact and account data.

Usage:
    python extract_quarterly_intel.py --company "Mondelez" --quarter "Q2" --year "2025"
    python extract_quarterly_intel.py --company "Mondelez" --quarter "Q2" --year "2025" --limit 25
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from difflib import get_close_matches

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey

from llm.ollama_client import OllamaClient
from ingestion.ingest_transcripts import EmbeddingProvider

# Import retail ontology components
from retail_ontology import (
    SYSTEM_PROMPT,
    JSON_SCHEMA_TEMPLATE,
    comprehensive_normalize_data,
    validate_extracted_data,
    generate_summary,
    calculate_confidence_score,
)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

load_dotenv()

COLLECTION_NAME = "RetailTranscriptChunk"
DEFAULT_LIMIT = 20
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

# Hierarchical retail category taxonomy
RETAIL_CATEGORY_TAXONOMY = {
    "Food & Beverage": {
        "Beverages": ["Coffee", "Tea", "Carbonated Soft Drinks", "Water", "Alcohol"],
        "Snacks & Confectionery": ["Chips", "Chocolate", "Candy", "Crackers", "Nuts"],
        "Pantry Staples": ["Spices", "Oils", "Baking", "Canned Goods", "Pasta"],
        "Dairy & Chilled": ["Milk", "Yogurt", "Cheese", "Butter"],
        "Frozen Foods": ["Ice Cream", "Frozen Meals", "Frozen Pizza"],
    },
    "Beauty & Personal Care": {
        "Hair Care": ["Shampoo", "Conditioner", "Styling", "Color"],
        "Skin Care": ["Face", "Body", "Sun Care", "Anti-Aging"],
        "Cosmetics": ["Makeup", "Nails", "Tools"],
        "Personal Hygiene": ["Deodorant", "Oral Care", "Shaving", "Bath & Body"],
    },
    "Home Care": {
        "Cleaning Supplies": ["Laundry", "Dishwashing", "Surface Cleaners"],
        "Paper Goods": ["Toilet Paper", "Paper Towels", "Tissues"],
    },
    "Health & Wellness": {
        "OTC Medication": ["Pain Relief", "Allergy", "Cold & Flu"],
        "Vitamins & Supplements": ["Multivitamins", "Protein Powder", "Minerals"],
    },
    "Baby & Child": {
        "Baby Care": ["Diapers", "Wipes", "Formula", "Baby Food"],
    },
    "Pet Care": {
        "Pet Food": ["Dog Food", "Cat Food", "Treats"],
    },
}

# Optimized system prompt for the LLM
SYSTEM_PROMPT = """You are an expert Retail Intelligence Analyst. 
Extract structured intelligence from earnings call transcripts.
Focus on:
1. Executive Leadership (names and titles)
2. Strategic priorities and focus areas
3. Retail Media investments and intent
4. Supply chain challenges and issues
5. Category strategy and growth focus

Return ONLY valid JSON. No explanations or markdown."""

# ============================================================================
# WEAVIATE OPERATIONS
# ============================================================================


def init_weaviate_client(logger: logging.Logger) -> weaviate.Client:
    """Initialize and connect to Weaviate database."""
    try:
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        auth = AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
        client = weaviate.Client(url=weaviate_url, auth_client_secret=auth)
        logger.info(f"✓ Connected to Weaviate at {weaviate_url}")
        return client
    except Exception as e:
        logger.error(f"✗ Error initializing Weaviate: {e}")
        raise


def get_all_companies_from_weaviate(
    client: weaviate.Client, logger: logging.Logger
) -> List[str]:
    """Retrieve all unique company names from Weaviate."""
    try:
        response = (
            client.query.get(COLLECTION_NAME, ["company"])
            .with_limit(1000)
            .do()
        )

        companies = set()
        for item in response["data"]["Get"][COLLECTION_NAME]:
            if item.get("company"):
                companies.add(item["company"])

        logger.debug(f"Found {len(companies)} unique companies in database")
        return sorted(list(companies))
    except Exception as e:
        logger.error(f"✗ Error retrieving companies from Weaviate: {e}")
        return []


def extract_company_from_query(
    query: str, available_companies: List[str], logger: logging.Logger
) -> Optional[str]:
    """Extract company name from query using fuzzy matching."""
    query_lower = query.lower()

    # Exact match first
    for company in available_companies:
        if company.lower() in query_lower:
            logger.debug(f"Exact company match found: {company}")
            return company

    # Fuzzy match
    query_words = set(re.findall(r"\b\w+\b", query_lower))
    for company in available_companies:
        company_lower = company.lower()
        company_parts = set(re.findall(r"\b\w+\b", company_lower))

        if company_parts & query_words:
            logger.debug(f"Fuzzy match found: {company}")
            return company

        for word in query_words:
            if len(word) > 3 and word in company_lower:
                logger.debug(f"Substring match found: {company}")
                return company

            if len(word) > 3:
                matches = get_close_matches(
                    word, [company_lower], n=1, cutoff=0.8
                )
                if matches:
                    logger.debug(f"Close match found: {company}")
                    return company

    logger.warning(f"Could not match company from query: {query}")
    return None


def query_weaviate(
    client: weaviate.Client,
    query_vector: List[float],
    company: Optional[str],
    quarter: Optional[str],
    year: Optional[str],
    limit: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Query Weaviate for transcript chunks."""
    try:
        # Build where conditions
        where_conditions = []

        if company:
            where_conditions.append({
                "path": ["company"],
                "operator": "Equal",
                "valueString": company,
            })

        if quarter:
            where_conditions.append({
                "path": ["quarter"],
                "operator": "Equal",
                "valueString": quarter,
            })

        if year:
            # Try as integer first (preferred), then as string
            try:
                year_int = int(year)
                where_conditions.append({
                    "path": ["year"],
                    "operator": "Equal",
                    "valueInt": year_int,
                })
            except (ValueError, TypeError):
                where_conditions.append({
                    "path": ["year"],
                    "operator": "Equal",
                    "valueString": year,
                })

        # Build query
        query_obj = (
            client.query
            .get(COLLECTION_NAME, ["text", "company", "quarter", "year", "page"])
            .with_near_vector({"vector": query_vector})
        )

        # Add where conditions if any
        if len(where_conditions) == 1:
            query_obj = query_obj.with_where(where_conditions[0])
        elif len(where_conditions) > 1:
            # Combine with AND
            combined = {
                "operator": "And",
                "operands": where_conditions,
            }
            query_obj = query_obj.with_where(combined)

        query_obj = query_obj.with_limit(limit)
        response = query_obj.do()

        chunks = response["data"]["Get"][COLLECTION_NAME]
        logger.info(f"✓ Retrieved {len(chunks)} chunks from Weaviate")
        return chunks

    except Exception as e:
        logger.error(f"✗ Error querying Weaviate: {e}")
        raise


# ============================================================================
# EMBEDDING & AGGREGATION
# ============================================================================


def embed_query(query: str, embedder: EmbeddingProvider, logger: logging.Logger) -> List[float]:
    """Embed the query."""
    try:
        vector = embedder.embed(query)
        logger.debug(f"✓ Query embedded (dimension: {len(vector)})")
        return vector
    except Exception as e:
        logger.error(f"✗ Error embedding query: {e}")
        raise


def aggregate_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Aggregate chunk texts into a single formatted string."""
    aggregated = []
    for i, chunk in enumerate(chunks, 1):
        company = chunk.get("company", "Unknown")
        quarter = chunk.get("quarter", "Unknown")
        year = chunk.get("year", "Unknown")
        page = chunk.get("page", "Unknown")
        text = chunk.get("text", "")

        header = f"[Chunk {i}] {company} - {quarter} {year} (Page {page})"
        aggregated.append(f"{header}\n{text}")

    return "\n\n".join(aggregated)


# ============================================================================
# LLM EXTRACTION
# ============================================================================


def extract_intelligence_from_text(
    text: str, logger: logging.Logger
) -> Dict[str, Any]:
    """Use LLM to extract structured intelligence from text using retail ontology."""
    
    client = OllamaClient(default_model="llama3.3:70b")

    # Simplified prompt with escaped curly braces to avoid LangChain template issues
    extraction_prompt = """Extract key retail intelligence from this earnings call transcript.

Focus on:
1. Executive leadership (names and titles)
2. Strategic priorities and commitments
3. Retail media investments and intent
4. Supply chain challenges and issues
5. Category strategy and focus areas
6. Risk factors and challenges

Return ONLY valid JSON in this format:
{{
  "leadership": [{{"name": "Full Name", "title": "Job Title"}}],
  "strategic_summary": "comprehensive summary of key strategic priorities and focus areas",
  "risk_summary": "comma-separated list of key risks (e.g., supply chain challenges, regulatory challenges, economic sensitivity, inflation)",
  "retail_media_flag": "true/false - does the company show intent for retail media investments?",
  "supply_chain_flag": "true/false - does the company have supply chain issues?",
  "category_flag": "true/false - does the company have category-specific focus?",
  "category_strategy": "specific retail categories like Beauty & Personal Care, Food & Beverage, Home Care, Health & Wellness (NOT strategic approaches like premiumization or digital commerce)",
  "executive_intent": {{
    "strategic_priorities": ["priority 1", "priority 2"]
  }}
}}

Use empty strings "" for fields not mentioned. Be specific and quote from the text.

TRANSCRIPT:
""" + text[:12000]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": extraction_prompt}
    ]

    try:
        response = client.chat(
            messages,
            model="llama3.3:70b",
            temperature=0.1,
        )

        # Clean response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Extract JSON if there's explanatory text
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            response = json_match.group(0)

        extracted = json.loads(response)
        
        # Use comprehensive normalization to map LLM creative fields to exact schema
        normalized_data = comprehensive_normalize_data(extracted)
        
        # Validate the normalized data
        if not validate_extracted_data(normalized_data):
            logger.warning("Schema validation failed after normalization")
        
        logger.info("✓ Intelligence extracted from LLM using retail ontology")
        return normalized_data

    except json.JSONDecodeError as e:
        logger.error(f"✗ LLM response is not valid JSON: {e}")
        logger.debug(f"Raw response: {response[:500]}")
        raise ValueError("Invalid JSON from LLM") from e
    except Exception as e:
        logger.error(f"✗ Error during LLM extraction: {e}")
        raise


# ============================================================================
# OUTPUT MAPPING
# ============================================================================


def map_to_output_schema(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Map comprehensive retail ontology data to the required output schema."""
    
    # contact_data_map: array of leaders
    contact_data_map = []
    leadership = extracted.get("leadership", []) or []
    for person in leadership:
        contact_data_map.append({
            "full_name": person.get("name", "").strip(),
            "job_title": person.get("title", "").strip(),
        })

    # Extract intelligence from the comprehensive ontology
    intelligence = extracted.get("intelligence", {}) or {}
    
    # Use LLM-generated strategic summary directly
    strategic_priority_summary = intelligence.get("strategic_summary", "").strip()
    
    # If no strategic summary, build from available data
    if not strategic_priority_summary:
        strategic_parts = []
        
        # Executive intent (highest priority)
        executive_intent = intelligence.get("executive_intent", {})
        if executive_intent.get("strategic_priorities"):
            priorities = executive_intent["strategic_priorities"]
            if isinstance(priorities, list) and priorities:
                strategic_parts.append(f"Priorities: {', '.join(priorities)}")
        
        # Category strategy
        if intelligence.get("category_strategy"):
            strategic_parts.append(f"Category Strategy: {intelligence['category_strategy']}")
        
        strategic_priority_summary = "; ".join(strategic_parts) if strategic_parts else ""
    
    # Use LLM-generated risk summary directly, with fallback to hardcoded logic
    risk_summary = intelligence.get("risk_summary", "").strip()
    
    # If LLM didn't provide a risk summary, fall back to hardcoded logic
    if not risk_summary:
        risk_parts = []
        
        if intelligence.get("supply_chain_shelf_pain_points"):
            risk_parts.append("supply chain challenges")
        
        risk_factors = intelligence.get("risk_factors", {})
        if risk_factors.get("supply_chain_vulnerabilities"):
            risk_parts.append("supply chain vulnerabilities")
        if risk_factors.get("regulatory_challenges"):
            risk_parts.append("regulatory challenges")
        if risk_factors.get("economic_sensitivity"):
            risk_parts.append("economic sensitivity")
        
        # Look for inflation mentions in various fields
        all_text = json.dumps(intelligence, default=str).lower()
        if "inflation" in all_text:
            risk_parts.append("inflation")
        
        risk_summary = ", ".join(risk_parts) if risk_parts else ""
    
    # Use LLM-generated boolean flags, with fallback to hardcoded logic
    retail_media_intent_flag = intelligence.get("retail_media_flag", "").strip().lower()
    if retail_media_intent_flag in ["true", "false"]:
        retail_media_intent_flag = retail_media_intent_flag == "true"
    else:
        # Fallback to hardcoded logic
        retail_media_intent_flag = bool(
            executive_intent.get("retail_media_budget") or
            intelligence.get("retail_media_investments")
        )
    
    supply_chain_issues_flag = intelligence.get("supply_chain_flag", "").strip().lower()
    if supply_chain_issues_flag in ["true", "false"]:
        supply_chain_issues_flag = supply_chain_issues_flag == "true"
    else:
        # Fallback to hardcoded logic
        risk_factors = intelligence.get("risk_factors", {})
        supply_chain_issues_flag = bool(
            intelligence.get("supply_chain_shelf_pain_points") or
            risk_factors.get("supply_chain_vulnerabilities")
        )
    
    category_focus_flag = intelligence.get("category_flag", "").strip().lower()
    if category_focus_flag in ["true", "false"]:
        category_focus_flag = category_focus_flag == "true"
    else:
        # Fallback to hardcoded logic
        category_focus_flag = bool(intelligence.get("category_strategy"))
    
    # Determine focussed category from category strategy
    focussed_category = ""
    category_text = intelligence.get("category_strategy", "").lower()
    
    # First try exact matches
    for category in FOCUSSED_CATEGORY_OPTIONS:
        if category.lower() in category_text:
            focussed_category = category
            break
    
    # If no exact match, try fuzzy keyword matching using taxonomy
    if not focussed_category:
        # Build comprehensive keyword mapping from taxonomy
        category_keywords = {}
        
        # Stop words to filter out
        stop_words = {'&', 'and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        
        for sector, categories in RETAIL_CATEGORY_TAXONOMY.items():
            for category, examples in categories.items():
                keywords = []
                # Add category name words (filter stop words)
                cat_words = [word for word in category.lower().split() if word not in stop_words]
                keywords.extend(cat_words)
                # Add example words (filter stop words and split multi-word examples)
                for example in examples:
                    example_words = example.lower().split()
                    keywords.extend([word for word in example_words if word not in stop_words])
                category_keywords[category] = list(set(keywords))  # Remove duplicates
        
        # Add some additional common keywords (more specific)
        category_keywords.update({
            "Personal Hygiene": ["beauty", "wellbeing", "personal", "hygiene", "grooming", "bath", "body", "deodorant", "oral", "shaving"],
            "Hair Care": ["hair", "shampoo", "conditioner", "styling", "color", "treatment"],
            "Skin Care": ["skin", "facial", "moisturizer", "lotion", "anti-aging", "sun", "body"],
            "Cosmetics": ["cosmetic", "makeup", "nails", "tools", "foundation", "lipstick"],
            "Cleaning Supplies": ["cleaning", "detergent", "laundry", "dishwashing", "surface", "household"],
            "Beverages": ["beverage", "drink", "coffee", "tea", "juice", "soft", "carbonated", "water", "alcohol"],
            "Snacks & Confectionery": ["snack", "confectionery", "candy", "chocolate", "chips", "nuts", "crackers"],
            "Dairy & Chilled": ["dairy", "milk", "cheese", "yogurt", "chilled", "butter", "cream"],
            "Frozen Foods": ["frozen", "ice", "cream", "freezer", "meals", "pizza", "vegetables"],
            "Baby Care": ["baby", "infant", "diaper", "wipes", "formula", "food", "toddler"],
            "Pet Food": ["pet", "animal", "dog", "cat", "treats", "food", "veterinary"],
            "OTC Medication": ["medication", "pharmacy", "pain", "relief", "allergy", "cold", "flu", "health"],
            "Vitamins & Supplements": ["vitamin", "supplement", "nutrition", "protein", "minerals", "multivitamin"],
        })
        
        # Score matches by specificity (prefer longer, more specific keywords)
        best_match = None
        best_score = 0
        
        for category, keywords in category_keywords.items():
            matches = [kw for kw in keywords if kw in category_text]
            if matches:
                # Score based on number of matches and keyword specificity
                score = len(matches) * sum(len(kw) for kw in matches)  # Prefer longer keywords
                if score > best_score:
                    best_score = score
                    best_match = category
        
        if best_match:
            focussed_category = best_match
    
    # Set category_focus_flag based on whether a focused category was actually identified
    category_focus_flag = bool(focussed_category)

    # account_data_map: strategic intelligence
    account_data_map = {
        "strategic_priority_summary": strategic_priority_summary,
        "risk_summary": risk_summary,
        "retail_media_intent_flag": retail_media_intent_flag,
        "supply_chain_issues_flag": supply_chain_issues_flag,
        "category_focus_flag": category_focus_flag,
        "focussed_category": focussed_category,
    }

    return {
        "contact_data_map": contact_data_map,
        "account_data_map": account_data_map,
    }


# ============================================================================
# MAIN QUERY FUNCTION
# ============================================================================


def query_intelligence(
    company: str,
    quarter: str,
    year: str,
    logger: Optional[logging.Logger] = None,
    limit: int = DEFAULT_LIMIT,
) -> Dict[str, Any]:
    """
    Query the database for a specific company, quarter, and year,
    then extract and return structured intelligence.

    Args:
        company: Company name (e.g., "Mondelez")
        quarter: Quarter (e.g., "Q2")
        year: Year (e.g., "2025")
        logger: Optional logger instance
        limit: Max chunks to retrieve

    Returns:
        Dict with contact_data_map and account_data_map

    Raises:
        ValueError: If query fails or no data found
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Querying: {company} {quarter} {year}")

    # Initialize clients
    weaviate_client = init_weaviate_client(logger)
    embedder = EmbeddingProvider(logger, True)

    # Build query string for embedding
    query_string = f"{company} {quarter} {year}"

    # Embed query
    query_vector = embed_query(query_string, embedder, logger)

    # Query Weaviate
    chunks = query_weaviate(
        weaviate_client,
        query_vector,
        company=company,
        quarter=quarter,
        year=year,
        limit=limit,
        logger=logger,
    )

    if not chunks:
        raise ValueError(
            f"No data found for {company} {quarter} {year} in database"
        )

    logger.info(f"Retrieved {len(chunks)} chunks")

    # Aggregate chunks
    aggregated_text = aggregate_chunks(chunks)

    # Extract intelligence using LLM
    extracted = extract_intelligence_from_text(aggregated_text, logger)

    # Map to output schema
    output = map_to_output_schema(extracted)
    output["_metadata"] = {
        "extraction_timestamp": datetime.now().isoformat(),
        "company": company,
        "quarter": quarter,
        "year": year,
        "chunks_used": len(chunks),
    }

    logger.info("✓ Intelligence extraction complete")
    return output


# ============================================================================
# MAIN CLI FUNCTION
# ============================================================================


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract retail intelligence from earnings call transcripts"
    )
    parser.add_argument(
        "--company",
        required=True,
        help="Company name (e.g., 'Mondelez')"
    )
    parser.add_argument(
        "--quarter",
        required=True,
        help="Quarter (e.g., 'Q2')"
    )
    parser.add_argument(
        "--year",
        required=True,
        help="Year (e.g., '2025')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum chunks to retrieve (default: {DEFAULT_LIMIT})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        result = query_intelligence(
            company=args.company,
            quarter=args.quarter,
            year=args.year,
            logger=logger,
            limit=args.limit,
        )
        
        # Save to file
        output_file = f"processed_intelligence/{args.company.lower()}_{args.quarter.lower()}_{args.year}.json"
        os.makedirs("processed_intelligence", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"✓ Results saved to {output_file}")
        
        # Also print to stdout
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()