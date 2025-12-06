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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from vault_client import load_vault_secrets
import weaviate
from weaviate.auth import AuthApiKey

from llm.ollama_client import OllamaClient
from ingestion.ingest_transcripts import EmbeddingProvider

# Import retail ontology components
from retail_ontology import (
    comprehensive_normalize_data,
    validate_extracted_data,
)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

load_vault_secrets()

COLLECTION_NAME = "RetailTranscriptChunk"
DEFAULT_LIMIT = 20

# ============================================================================
# LLM EXTRACTION
# ============================================================================


def init_weaviate_client(logger: logging.Logger) -> weaviate.Client:
    """Initialize and connect to Weaviate database."""
    try:
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        if not weaviate_url:
            raise ValueError("WEAVIATE_URL environment variable is not set")

        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        auth = AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None

        client = weaviate.Client(url=weaviate_url, auth_client_secret=auth)
        logger.info(f"✓ Connected to Weaviate at {weaviate_url}")
        return client
    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Failed to connect to Weaviate database: {e}")
        raise ConnectionError(f"Database connection failed: {e}") from e


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
        # Validate inputs
        if not query_vector or len(query_vector) == 0:
            raise ValueError("Query vector cannot be empty")
        if limit <= 0:
            raise ValueError("Limit must be a positive integer")

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
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse year as integer, using as string: {e}")
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

        # Execute query
        response = query_obj.do()

        # Validate response structure
        if not response or "data" not in response or "Get" not in response["data"]:
            raise ValueError("Invalid response structure from Weaviate")

        chunks = response["data"]["Get"][COLLECTION_NAME]
        if not isinstance(chunks, list):
            raise ValueError("Chunks data is not a list")

        logger.info(f"✓ Retrieved {len(chunks)} chunks from Weaviate")
        return chunks

    except ValueError as e:
        logger.error(f"✗ Validation error in query: {e}")
        raise
    except KeyError as e:
        logger.error(f"✗ Unexpected response structure from Weaviate: {e}")
        raise RuntimeError(f"Database returned unexpected data format: {e}") from e
    except Exception as e:
        logger.error(f"✗ Database query failed: {e}")
        raise RuntimeError(f"Failed to query database: {e}") from e


# ============================================================================
# EMBEDDING & AGGREGATION
# ============================================================================


def embed_query(query: str, embedder: EmbeddingProvider, logger: logging.Logger) -> List[float]:
    """Embed the query."""
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        if not embedder:
            raise ValueError("Embedding provider not available")

        vector = embedder.embed(query.strip())
        if not vector or len(vector) == 0:
            raise RuntimeError("Embedding provider returned empty vector")

        logger.debug(f"✓ Query embedded (dimension: {len(vector)})")
        return vector

    except ValueError as e:
        logger.error(f"✗ Validation error in embedding: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error embedding query: {e}")
        raise RuntimeError(f"Failed to generate embedding: {e}") from e


def aggregate_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Aggregate chunk texts into a single formatted string."""
    try:
        if not chunks:
            raise ValueError("No chunks provided for aggregation")

        if not isinstance(chunks, list):
            raise ValueError("Chunks must be a list")

        aggregated = []
        for i, chunk in enumerate(chunks, 1):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk {i} is not a dictionary")

            company = chunk.get("company", "Unknown")
            quarter = chunk.get("quarter", "Unknown")
            year = chunk.get("year", "Unknown")
            page = chunk.get("page", "Unknown")
            text = chunk.get("text", "")

            if not text or not text.strip():
                raise ValueError(f"Chunk {i} has empty or missing text content")

            header = f"[Chunk {i}] {company} - {quarter} {year} (Page {page})"
            aggregated.append(f"{header}\n{text.strip()}")

        if not aggregated:
            raise ValueError("No valid chunks found after processing")

        return "\n\n".join(aggregated)

    except ValueError as e:
        raise  # Re-raise validation errors
    except Exception as e:
        raise RuntimeError(f"Failed to aggregate chunks: {e}") from e


# ============================================================================
# LLM EXTRACTION
# ============================================================================


def extract_intelligence_from_text(
    text: str, logger: logging.Logger
) -> Dict[str, Any]:
    """Use LLM to extract structured intelligence from text using retail ontology."""
    try:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if not logger:
            raise ValueError("Logger is required")

        client = OllamaClient(default_model="llama3.3:70b")
        if not client:
            raise RuntimeError("Failed to initialize Ollama client")

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
  "risk_summary": "bullet-point list of risks from the following categories that are mentioned or implied in the transcript: Supply chain disruptions and logistics challenges, Regulatory changes and compliance requirements, Economic pressures and inflation, Competitive dynamics and market shifts, Consumer behavior changes, Raw material costs and availability, Currency fluctuations, Geopolitical events. Do not include categories that are not mentioned or implied.",
  "retail_media_flag": "true/false - does the company show intent for retail media investments?",
  "supply_chain_flag": "true/false - does the company have supply chain issues?",
  "category_flag": "true/false - does the company have category-specific focus?",
  "category_strategy": "specific retail categories like Beauty & Personal Care, Food & Beverage, Home Care, Health & Wellness (NOT strategic approaches like premiumization or digital commerce)",
  "focussed_category": ["array of specific retail categories the company is focusing on, based on the transcript content"],
  "executive_intent": {{
    "strategic_priorities": ["priority 1", "priority 2"]
  }}
}}

Use empty strings "" for fields not mentioned. Be specific and quote from the text.

TRANSCRIPT:
""" + text[:12000]

        messages = [
            {"role": "user", "content": extraction_prompt}
        ]

        response = client.chat(
            messages,
            model="llama3.3:70b",
            temperature=0.1,
        )

        if not response or not response.strip():
            raise RuntimeError("LLM returned empty response")

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

        if not response:
            raise RuntimeError("No JSON found in LLM response")

        extracted = json.loads(response)

        if not isinstance(extracted, dict):
            raise ValueError("LLM response is not a JSON object")

        # Use comprehensive normalization to map LLM creative fields to exact schema
        normalized_data = comprehensive_normalize_data(extracted)

        # Validate the normalized data
        if not validate_extracted_data(normalized_data):
            logger.warning("Schema validation failed after normalization")

        logger.info("✓ Intelligence extracted from LLM using retail ontology")
        return normalized_data

    except ValueError as e:
        logger.error(f"✗ Validation error in LLM extraction: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"✗ LLM response is not valid JSON: {e}")
        logger.debug(f"Raw response: {response[:500]}")
        raise ValueError("Invalid JSON from LLM") from e
    except RuntimeError as e:
        logger.error(f"✗ Runtime error in LLM extraction: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error during LLM extraction: {e}")
        raise RuntimeError(f"LLM extraction failed: {e}") from e


# ============================================================================
# OUTPUT MAPPING
# ============================================================================


def map_to_output_schema(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Map comprehensive retail ontology data to the required output schema."""
    try:
        if not extracted or not isinstance(extracted, dict):
            raise ValueError("Extracted data must be a non-empty dictionary")

        # contact_data_map: array of leaders
        contact_data_map = []
        leadership = extracted.get("leadership", []) or []
        if not isinstance(leadership, list):
            raise ValueError("Leadership data must be a list")

        for person in leadership:
            if not isinstance(person, dict):
                continue  # Skip invalid entries
            contact_data_map.append({
                "full_name": person.get("name", "").strip(),
                "job_title": person.get("title", "").strip(),
            })

        # Extract intelligence from the comprehensive ontology
        intelligence = extracted.get("intelligence", {}) or {}
        if not isinstance(intelligence, dict):
            intelligence = {}

        # Executive intent (used in multiple places)
        executive_intent = intelligence.get("executive_intent", {})

        # Use LLM-generated strategic summary directly
        strategic_priority_summary = intelligence.get("strategic_summary", "").strip()

        # If no strategic summary, build from available data
        if not strategic_priority_summary:
            strategic_parts = []

            # Executive intent (highest priority)
            if isinstance(executive_intent, dict) and executive_intent.get("strategic_priorities"):
                priorities = executive_intent["strategic_priorities"]
                if isinstance(priorities, list) and priorities:
                    strategic_parts.append(f"Priorities: {', '.join(priorities)}")

            # Category strategy
            if intelligence.get("category_strategy"):
                strategic_parts.append(f"Category Strategy: {intelligence['category_strategy']}")

            strategic_priority_summary = "; ".join(strategic_parts) if strategic_parts else ""

        # Use LLM-generated risk summary directly
        risk_summary = intelligence.get("risk_summary", "")
        if isinstance(risk_summary, list):
            # If LLM returns a list, join it with newlines
            risk_summary = "\n".join(f"• {item}" if not item.startswith("•") else item for item in risk_summary)
        elif isinstance(risk_summary, str):
            risk_summary = risk_summary.strip()
        else:
            risk_summary = ""

        # Use LLM-generated boolean flags, with fallback to hardcoded logic
        retail_media_intent_flag = intelligence.get("retail_media_flag", "")
        if isinstance(retail_media_intent_flag, str):
            retail_media_intent_flag = retail_media_intent_flag.strip().lower()
            if retail_media_intent_flag in ["true", "false"]:
                retail_media_intent_flag = retail_media_intent_flag == "true"
            else:
                # Fallback to hardcoded logic
                retail_media_intent_flag = bool(
                    executive_intent.get("retail_media_budget") or
                    intelligence.get("retail_media_investments")
                )
        elif isinstance(retail_media_intent_flag, bool):
            # LLM returned boolean directly
            pass
        else:
            # Fallback to hardcoded logic
            retail_media_intent_flag = bool(
                executive_intent.get("retail_media_budget") or
                intelligence.get("retail_media_investments")
            )

        supply_chain_issues_flag = intelligence.get("supply_chain_flag", "")
        if isinstance(supply_chain_issues_flag, str):
            supply_chain_issues_flag = supply_chain_issues_flag.strip().lower()
            if supply_chain_issues_flag in ["true", "false"]:
                supply_chain_issues_flag = supply_chain_issues_flag == "true"
            else:
                # Fallback to hardcoded logic
                risk_factors = intelligence.get("risk_factors", {})
                supply_chain_issues_flag = bool(
                    intelligence.get("supply_chain_shelf_pain_points") or
                    risk_factors.get("supply_chain_vulnerabilities")
                )
        elif isinstance(supply_chain_issues_flag, bool):
            # LLM returned boolean directly
            pass
        else:
            # Fallback to hardcoded logic
            risk_factors = intelligence.get("risk_factors", {})
            supply_chain_issues_flag = bool(
                intelligence.get("supply_chain_shelf_pain_points") or
                risk_factors.get("supply_chain_vulnerabilities")
            )

        category_focus_flag = intelligence.get("category_flag", "")
        if isinstance(category_focus_flag, str):
            category_focus_flag = category_focus_flag.strip().lower()
            if category_focus_flag in ["true", "false"]:
                category_focus_flag = category_focus_flag == "true"
            else:
                # Fallback to hardcoded logic
                category_focus_flag = bool(intelligence.get("category_strategy"))
        elif isinstance(category_focus_flag, bool):
            # LLM returned boolean directly
            pass
        else:
            # Fallback to hardcoded logic
            category_focus_flag = bool(intelligence.get("category_strategy"))

        # Determine focussed categories from category strategy
        focussed_categories = []
        category_text = intelligence.get("category_strategy", "").lower()

        # Check if LLM directly provided focussed_category as array
        llm_categories = intelligence.get("focussed_category", [])
        if isinstance(llm_categories, list) and llm_categories:
            focussed_categories = [cat.strip() for cat in llm_categories if cat.strip()]
        else:
            # Fallback: Hierarchical category mapping based on retail structure
            category_hierarchy = {
                # Food & Beverage
                "Beverages": ["coffee", "tea", "carbonated soft drinks", "water", "alcohol", "beverage", "drink", "soda", "juice", "energy drinks"],
                "Snacks & Confectionery": ["chips", "chocolate", "candy", "crackers", "nuts", "snack", "confectionery", "cookies", "biscuits", "pretzels"],
                "Pantry Staples": ["spices", "oils", "baking", "canned goods", "pasta", "rice", "flour", "sugar", "salt", "condiments"],
                "Dairy & Chilled": ["milk", "yogurt", "cheese", "butter", "cream", "dairy", "chilled", "eggs", "margarine"],
                "Frozen Foods": ["ice cream", "frozen meals", "frozen pizza", "frozen vegetables", "frozen", "ice cream", "popsicles"],

                # Beauty & Personal Care
                "Hair Care": ["shampoo", "conditioner", "styling", "color", "hair", "treatment", "hair dye", "gel", "spray"],
                "Skin Care": ["face", "body", "sun care", "anti-aging", "skin", "moisturizer", "lotion", "cream", "serum", "cleanser"],
                "Cosmetics": ["makeup", "nails", "tools", "cosmetic", "foundation", "lipstick", "mascara", "eyeshadow", "blush"],
                "Personal Hygiene": ["deodorant", "oral care", "shaving", "bath & body", "beauty", "wellbeing", "personal", "hygiene", "toothpaste", "soap", "shower gel"],

                # Home Care
                "Cleaning Supplies": ["laundry", "dishwashing", "surface cleaners", "cleaning", "detergent", "bleach", "disinfectant"],
                "Paper Goods": ["toilet paper", "paper towels", "tissues", "napkins", "paper", "facial tissue"],

                # Health & Wellness
                "OTC Medication": ["pain relief", "allergy", "cold & flu", "medication", "pharmacy", "pain", "headache", "fever"],
                "Vitamins & Supplements": ["multivitamins", "protein powder", "minerals", "vitamin", "supplement", "nutrition", "dietary"],

                # Baby & Child
                "Baby Care": ["diapers", "wipes", "formula", "baby food", "baby", "infant", "diaper", "baby formula", "baby milk"],

                # Pet Care
                "Pet Food": ["dog food", "cat food", "treats", "pet", "animal", "dog", "cat", "pet food", "kibble"]
            }

            for category, keywords in category_hierarchy.items():
                if any(keyword in category_text for keyword in keywords):
                    focussed_categories.append(category)

        # Set category_focus_flag based on whether any focused categories were identified
        category_focus_flag = bool(focussed_categories)

        # account_data_map: strategic intelligence
        account_data_map = {
            "strategic_priority_summary": strategic_priority_summary,
            "risk_summary": risk_summary,
            "retail_media_intent_flag": retail_media_intent_flag,
            "supply_chain_issues_flag": supply_chain_issues_flag,
            "category_focus_flag": category_focus_flag,
            "focussed_category": focussed_categories,
        }

        return {
            "contact_data_map": contact_data_map,
            "account_data_map": account_data_map,
        }

    except ValueError as e:
        raise  # Re-raise validation errors
    except Exception as e:
        raise RuntimeError(f"Failed to map data to output schema: {e}") from e


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
    try:
        # Validate inputs
        if not company or not company.strip():
            raise ValueError("Company name cannot be empty")
        if not quarter or not quarter.strip():
            raise ValueError("Quarter cannot be empty")
        if not year or not year.strip():
            raise ValueError("Year cannot be empty")
        if limit <= 0:
            raise ValueError("Limit must be a positive integer")

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

    except ValueError as e:
        logger.error(f"✗ Validation error in query: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"✗ Runtime error in query: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error in query intelligence: {e}")
        raise RuntimeError(f"Intelligence extraction failed: {e}") from e


# ============================================================================
# MAIN CLI FUNCTION
# ============================================================================


def main():
    """Main CLI entry point."""
    try:
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
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Results saved to {output_file}")

        # Also print to stdout
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()