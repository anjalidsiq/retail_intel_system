"""
Comprehensive Delta Analysis: Multi-Dimensional Comparative Analysis

This script performs deep comparative analysis between periods focusing on:
- Previous Quarter (QoQ) analysis
- Same Quarter Previous Year (YoY) analysis

Using retail domain ontology for comprehensive insights.

Usage:
    python comprehensive_delta_analysis.py --company "Unilever" --quarter "Q3" --year 2025
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import weaviate
from vault_client import load_vault_secrets
from llm.ollama_client import OllamaClient
from retail_ontology import RETAIL_CONCEPTS, SYSTEM_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_vault_secrets()

class ComprehensiveDeltaAnalyzer:
    """Comprehensive multi-dimensional delta analysis using retail ontology and Weaviate context."""

    def __init__(self, model_name: str = "llama3.3:70b"):
        self.llm_client = OllamaClient(default_model=model_name)
        self.weaviate_client = None
        self.retail_concepts = RETAIL_CONCEPTS
        self.concept_categories = list(self.retail_concepts.keys())
        self.collection_name = "RetailTranscriptChunk"
        
        # Get Weaviate URL from environment (vault) or use fallback
        weaviate_url = os.getenv("WEAVIATE_URL", "http://172.16.9.28:8080")
        
        # Try to connect to Weaviate, but continue if it fails
        try:
            self.weaviate_client = weaviate.Client(weaviate_url, startup_period=10)
            logger.info(f"✅ Connected to Weaviate at {weaviate_url}")
        except Exception as e:
            logger.warning(f"⚠️  Weaviate connection failed: {e}. Analysis will use JSON only.")

    def query_weaviate_for_period(
        self, 
        company: str, 
        quarter: str, 
        year: int,
        limit: int = 20
    ) -> List[Dict]:
        """Query Weaviate for chunks from specific period."""
        if not self.weaviate_client:
            logger.warning("Weaviate client not available, skipping chunk retrieval")
            return []
            
        try:
            where_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["company"],
                        "operator": "Equal",
                        "valueText": company
                    },
                    {
                        "path": ["quarter"],
                        "operator": "Equal",
                        "valueText": quarter
                    },
                    {
                        "path": ["year"],
                        "operator": "Equal",
                        "valueInt": year
                    }
                ]
            }

            response = self.weaviate_client.query.get(
                self.collection_name,
                ["text", "page", "concept_hits", "concept_score"]
            ).with_where(where_filter).with_limit(limit).do()

            chunks = response.get("data", {}).get("Get", {}).get(self.collection_name, [])
            logger.info(f"Retrieved {len(chunks)} chunks for {company} {quarter} {year}")
            return chunks

        except Exception as e:
            logger.warning(f"Error querying Weaviate: {e}")
            return []

    def extract_key_quotes(self, chunks: List[Dict], num_quotes: int = 3) -> List[str]:
        """Extract most relevant quotes from chunks."""
        # Sort by concept score (higher = more relevant)
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: x.get("concept_score", 0), 
            reverse=True
        )
        
        quotes = []
        for chunk in sorted_chunks[:num_quotes]:
            text = chunk.get("text", "").strip()
            if text and len(text) > 50:  # Filter out short chunks
                # Truncate to 250 chars for readability
                truncated = text[:250] + ("..." if len(text) > 250 else "")
                quotes.append(f'"{truncated}"')
        
        return quotes

    def load_processed_intelligence(self, company: str, quarter: str, year: int) -> Optional[Dict]:
        """Load processed intelligence JSON for given parameters."""
        filename = f"{company.lower()}_{quarter.lower()}_{year}.json"
        filepath = os.path.join("processed_intelligence", filename)
        
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None

    def get_previous_quarter(self, quarter: str, year: int) -> tuple[str, int]:
        """Get previous quarter and year."""
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        current_idx = quarters.index(quarter)

        if current_idx == 0:
            return "Q4", year - 1
        else:
            return quarters[current_idx - 1], year

    def extract_intelligence_fields(self, json_data: Dict) -> Dict[str, str]:
        """Extract all intelligence-related fields from processed JSON."""
        if not json_data:
            return {}
        
        account_data = json_data.get("account_data_map", {})
        
        return {
            "strategic_priority_summary": account_data.get("strategic_priority_summary", ""),
            "risk_summary": account_data.get("risk_summary", ""),
            "retail_media_intent_flag": account_data.get("retail_media_intent_flag", False),
            "supply_chain_issues_flag": account_data.get("supply_chain_issues_flag", False),
            "category_focus_flag": account_data.get("category_focus_flag", False),
            "focussed_category": account_data.get("focussed_category", []),
        }

    def extract_concept_mentions(self, text: str) -> Dict[str, List[str]]:
        """Extract retail concepts mentioned in text using ontology."""
        concepts_found = {}
        
        for category, terms in self.retail_concepts.items():
            found_terms = []
            text_lower = text.lower()
            
            for term in terms:
                if term.lower() in text_lower:
                    found_terms.append(term)
            
            if found_terms:
                concepts_found[category] = found_terms
        
        return concepts_found

    def analyze_flag_changes(self, current_flags: Dict, previous_flags: Dict) -> Dict[str, str]:
        """Analyze changes in strategic flags between periods."""
        changes = {
            "retail_media_focus": "No change",
            "supply_chain_focus": "No change",
            "category_focus": "No change"
        }
        
        if current_flags.get("retail_media_intent_flag") != previous_flags.get("retail_media_intent_flag"):
            current_val = current_flags.get("retail_media_intent_flag", False)
            changes["retail_media_focus"] = f"Shifted to {'ACTIVE' if current_val else 'INACTIVE'}"
        
        if current_flags.get("supply_chain_issues_flag") != previous_flags.get("supply_chain_issues_flag"):
            current_val = current_flags.get("supply_chain_issues_flag", False)
            changes["supply_chain_focus"] = f"Shifted to {'ACTIVE' if current_val else 'INACTIVE'}"
        
        if current_flags.get("category_focus_flag") != previous_flags.get("category_focus_flag"):
            current_val = current_flags.get("category_focus_flag", False)
            changes["category_focus"] = f"Shifted to {'ACTIVE' if current_val else 'INACTIVE'}"
            
            # Check category changes
            current_cats = set(current_flags.get("focussed_category", []))
            previous_cats = set(previous_flags.get("focussed_category", []))
            
            new_cats = current_cats - previous_cats
            removed_cats = previous_cats - current_cats
            
            if new_cats:
                changes["category_focus"] += f" | New: {', '.join(new_cats)}"
            if removed_cats:
                changes["category_focus"] += f" | Removed: {', '.join(removed_cats)}"
        
        return changes

    def build_comprehensive_prompt(self, current_json: Dict, current_chunks: List[Dict],
                                  qoq_json: Optional[Dict], qoq_chunks: List[Dict],
                                  yoy_json: Optional[Dict], yoy_chunks: List[Dict],
                                  analysis_type: str,
                                  current_period: str = None,
                                  comparison_period: str = None) -> str:
        """Build comprehensive analysis prompt with JSON + Weaviate context."""
        
        current_intel = self.extract_intelligence_fields(current_json)
        qoq_intel = self.extract_intelligence_fields(qoq_json) if qoq_json else {}
        yoy_intel = self.extract_intelligence_fields(yoy_json) if yoy_json else {}
        
        # Extract concepts
        current_concepts = self.extract_concept_mentions(
            current_intel.get("strategic_priority_summary", "") + " " + 
            current_intel.get("risk_summary", "")
        )
        qoq_concepts = self.extract_concept_mentions(
            qoq_intel.get("strategic_priority_summary", "") + " " + 
            qoq_intel.get("risk_summary", "")
        ) if qoq_json else {}
        yoy_concepts = self.extract_concept_mentions(
            yoy_intel.get("strategic_priority_summary", "") + " " + 
            yoy_intel.get("risk_summary", "")
        ) if yoy_json else {}
        
        # Analyze flag changes
        qoq_flag_changes = self.analyze_flag_changes(current_intel, qoq_intel) if qoq_json else {}
        yoy_flag_changes = self.analyze_flag_changes(current_intel, yoy_intel) if yoy_json else {}
        
        # Extract quotes from Weaviate chunks
        current_quotes = self.extract_key_quotes(current_chunks, num_quotes=3)
        qoq_quotes = self.extract_key_quotes(qoq_chunks, num_quotes=3) if qoq_chunks else []
        yoy_quotes = self.extract_key_quotes(yoy_chunks, num_quotes=3) if yoy_chunks else []
        
        # Convert to string formats safely - escape curly braces for LangChain
        qoq_concepts_str = str(qoq_concepts).replace("{", "{{").replace("}", "}}") if qoq_concepts else 'No data'
        current_concepts_str = str(current_concepts).replace("{", "{{").replace("}", "}}")
        qoq_flag_changes_str = str(qoq_flag_changes).replace("{", "{{").replace("}", "}}") if qoq_flag_changes else 'No changes'
        yoy_concepts_str = str(yoy_concepts).replace("{", "{{").replace("}", "}}") if yoy_concepts else 'No data'
        yoy_flag_changes_str = str(yoy_flag_changes).replace("{", "{{").replace("}", "}}") if yoy_flag_changes else 'No changes'

        if analysis_type == "qoq":
            comparison_context = f"""
Quarter-over-Quarter (QoQ) Analysis:
Previous Quarter: {comparison_period if comparison_period else 'N/A'}
Current Quarter: {current_period if current_period else 'N/A'}

Previous Quarter Concepts: {qoq_concepts_str}
Current Quarter Concepts: {current_concepts_str}

Strategic Flag Changes (QoQ): {qoq_flag_changes_str}

Previous Quarter Strategic Priority: {qoq_intel.get('strategic_priority_summary', 'N/A')[:500]}
Current Quarter Strategic Priority: {current_intel.get('strategic_priority_summary', '')[:500]}

Previous Quarter Risks: {qoq_intel.get('risk_summary', 'N/A')[:500]}
Current Quarter Risks: {current_intel.get('risk_summary', '')[:500]}

Previous Quarter Direct Quotes from Transcript:
{chr(10).join(qoq_quotes) if qoq_quotes else 'No quotes available'}

Current Quarter Direct Quotes from Transcript:
{chr(10).join(current_quotes) if current_quotes else 'No quotes available'}
"""
        else:  # yoy
            comparison_context = f"""
Year-over-Year (YoY) Analysis:
Same Quarter Previous Year: {comparison_period if comparison_period else 'N/A'}
Current Quarter: {current_period if current_period else 'N/A'}

Previous Year Concepts: {yoy_concepts_str}
Current Year Concepts: {current_concepts_str}

Strategic Flag Changes (YoY): {yoy_flag_changes_str}

Previous Year Strategic Priority: {yoy_intel.get('strategic_priority_summary', 'N/A')[:500]}
Current Year Strategic Priority: {current_intel.get('strategic_priority_summary', '')[:500]}

Previous Year Risks: {yoy_intel.get('risk_summary', 'N/A')[:500]}
Current Year Risks: {current_intel.get('risk_summary', '')[:500]}

Previous Year Direct Quotes from Transcript:
{chr(10).join(yoy_quotes) if yoy_quotes else 'No quotes available'}

Current Year Direct Quotes from Transcript:
{chr(10).join(current_quotes) if current_quotes else 'No quotes available'}
"""

        prompt = f"""You are a retail intelligence analyst specializing in:
- Consumer packaged goods (CPG) strategies
- Retail media networks and e-commerce
- Supply chain and digital shelf optimization
- Channel strategy and direct-to-consumer (DTC)
- Financial performance and strategic priorities
- Competitive positioning and market dynamics
- Product innovation and premiumization
- Strategic partnerships and M&A activity

{comparison_context}

**Comprehensive Analysis Dimensions:**

1. **Strategic Pivots**: What topics, themes, or priorities have fundamentally changed? Identify:
   - Emerging focus areas not mentioned before
   - Abandoned or de-emphasized topics
   - Intensity changes in existing themes

2. **Risk Perception Evolution**: How has risk sentiment and focus changed?
   - New risks identified
   - Previous risks resolved or downplayed
   - Severity assessment shifts
   - Geographic/category risk variations

3. **Retail Media & AdTech Strategy**: Changes in advertising and retail media focus
   - Investment level changes
   - Platform strategy shifts
   - Ad revenue expectations
   - First-party data and clean room initiatives

4. **Supply Chain & Digital Shelf**: Evolution of operational strategy
   - OOS and inventory management focus
   - Supply chain resilience improvements
   - Margin pressure management
   - Logistics and fulfillment changes

5. **Channel & DTC Strategy**: Distribution and direct-to-consumer evolution
   - Omnichannel investments
   - Marketplace presence and strategy
   - DTC capability development
   - Click-and-collect or hybrid models

6. **Premiumization & Product Strategy**: Product portfolio and pricing changes
   - Premium product mix shifts
   - SKU rationalization progress
   - Price architecture changes
   - Category performance shifts

7. **Financial Priorities & KPIs**: Financial target and performance shifts
   - Growth rate changes
   - Margin target adjustments
   - Cash flow priorities
   - Investment allocation changes

8. **Competitive Positioning**: Market dynamics and competitive stance
   - Market share trends and goals
   - Competitive intensity perception
   - Brand equity initiatives
   - Category consolidation or fragmentation views

9. **Technology & Digital Transformation**: Technology investment and capability building
   - E-commerce platform investments
   - Data platform (CDP) initiatives
   - AI/automation implementations
   - Digital marketing ROI focus

10. **M&A & Partnerships**: Strategic alliances and acquisition strategy
    - M&A activity or plans
    - Strategic partnerships and collaborations
    - Retailer relationship evolution
    - Technology vendor partnerships

**Output Format**: Return a JSON object with these keys: "strategic_pivots", "risk_perception", "retail_media_strategy", 
"supply_chain_strategy", "channel_strategy", "premiumization_strategy", "financial_priorities", "competitive_positioning", 
"technology_innovation", "partnerships_ma". Each key should contain a comprehensive analysis text addressing that dimension.

**Analysis Style**:
- Be specific and quantitative where possible
- Reference actual data points from the transcripts
- Highlight magnitude of changes (major/moderate/minor)
- Note any contradictions or nuances
- Provide business implications and significance
"""
        return prompt

    def perform_comprehensive_analysis(self, current_json: Dict, current_chunks: List[Dict],
                                      comparison_json: Optional[Dict], comparison_chunks: List[Dict],
                                      analysis_type: str,
                                      current_period: str = None, comparison_period: str = None) -> Dict[str, str]:
        """Perform comprehensive multi-dimensional analysis with Weaviate context."""
        
        if not comparison_json:
            comparison_json = {}
        if not comparison_chunks:
            comparison_chunks = []
        
        prompt = self.build_comprehensive_prompt(
            current_json, current_chunks,
            comparison_json if analysis_type == "qoq" else None, comparison_chunks if analysis_type == "qoq" else [],
            comparison_json if analysis_type == "yoy" else None, comparison_chunks if analysis_type == "yoy" else [],
            analysis_type, current_period, comparison_period
        )

        try:
            response = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.15,
                max_tokens=2000
            )
            response = response.strip()
            
            # Extract JSON from response
            if '```json' in response:
                start_idx = response.find('```json') + 7
                end_idx = response.find('```', start_idx)
                json_str = response[start_idx:end_idx]
            elif '```' in response:
                start_idx = response.find('```') + 3
                end_idx = response.find('```', start_idx)
                json_str = response[start_idx:end_idx]
            else:
                json_str = response
            
            json_str = json_str.strip()
            
            try:
                parsed_response = json.loads(json_str)
                return {
                    "Strategic Pivots:": parsed_response.get("strategic_pivots", ""),
                    "Risk Perception:": parsed_response.get("risk_perception", ""),
                    "Retail Media Strategy:": parsed_response.get("retail_media_strategy", ""),
                    "Supply Chain Strategy:": parsed_response.get("supply_chain_strategy", ""),
                    "Channel Strategy:": parsed_response.get("channel_strategy", ""),
                    "Premiumization Strategy:": parsed_response.get("premiumization_strategy", ""),
                    "Financial Priorities:": parsed_response.get("financial_priorities", ""),
                    "Competitive Positioning:": parsed_response.get("competitive_positioning", ""),
                    "Technology Innovation:": parsed_response.get("technology_innovation", ""),
                    "Partnerships & M&A:": parsed_response.get("partnerships_ma", "")
                }
            except json.JSONDecodeError:
                return self._parse_fallback_response(response)
                
        except Exception as e:
            error_msg = f"Error generating analysis: {e}"
            return {
                "Strategic Pivots:": error_msg,
                "Risk Perception:": error_msg,
                "Retail Media Strategy:": error_msg,
                "Supply Chain Strategy:": error_msg,
                "Channel Strategy:": error_msg,
                "Premiumization Strategy:": error_msg,
                "Financial Priorities:": error_msg,
                "Competitive Positioning:": error_msg,
                "Technology Innovation:": error_msg,
                "Partnerships & M&A:": error_msg
            }

    def _parse_fallback_response(self, response: str) -> Dict[str, str]:
        """Fallback parser if LLM doesn't return proper JSON."""
        result = {
            "Strategic Pivots:": "",
            "Risk Perception:": "",
            "Retail Media Strategy:": "",
            "Supply Chain Strategy:": "",
            "Channel Strategy:": "",
            "Premiumization Strategy:": "",
            "Financial Priorities:": "",
            "Competitive Positioning:": "",
            "Technology Innovation:": "",
            "Partnerships & M&A:": ""
        }
        
        response_lower = response.lower()
        keys_to_find = ["strategic pivots", "risk perception", "retail media strategy", 
                       "supply chain strategy", "channel strategy", "premiumization strategy",
                       "financial priorities", "competitive positioning", "technology innovation", 
                       "partnerships & m&a"]
        
        for i, key in enumerate(keys_to_find):
            if key in response_lower:
                start_idx = response_lower.find(key)
                next_key_idx = len(response)
                
                for next_key in keys_to_find[i+1:]:
                    if next_key in response_lower[start_idx + 1:]:
                        next_key_idx = min(next_key_idx, response_lower.find(next_key, start_idx + 1))
                
                result[key.title() + ":"] = response[start_idx:next_key_idx].strip()
        
        return result

    def analyze_company(self, company: str, quarter: str, year: int) -> Dict:
        """Main analysis function performing both QoQ and YoY comparisons with Weaviate enrichment."""

        logger.info(f"Starting comprehensive analysis for {company} {quarter} {year}")
        
        # Load JSONs
        current_json = self.load_processed_intelligence(company, quarter, year)
        if not current_json:
            return {"error": f"No processed intelligence found for {company} {quarter} {year}"}

        # Load previous quarter (QoQ)
        prev_quarter, prev_year = self.get_previous_quarter(quarter, year)
        qoq_json = self.load_processed_intelligence(company, prev_quarter, prev_year)

        # Load same quarter last year (YoY)
        yoy_json = self.load_processed_intelligence(company, quarter, year - 1)

        # Query Weaviate for transcript chunks
        logger.info("Querying Weaviate for transcript chunks...")
        current_chunks = self.query_weaviate_for_period(company, quarter, year)
        qoq_chunks = self.query_weaviate_for_period(company, prev_quarter, prev_year) if qoq_json else []
        yoy_chunks = self.query_weaviate_for_period(company, quarter, year - 1) if yoy_json else []
        
        logger.info(f"Current chunks: {len(current_chunks)}, QoQ: {len(qoq_chunks)}, YoY: {len(yoy_chunks)}")

        # Perform analyses with Weaviate context
        current_period_str = f"{quarter} {year}"
        qoq_period_str = f"{prev_quarter} {prev_year}"
        yoy_period_str = f"{quarter} {year-1}"
        
        qoq_analysis = self.perform_comprehensive_analysis(
            current_json, current_chunks, qoq_json, qoq_chunks, "qoq",
            current_period=current_period_str,
            comparison_period=qoq_period_str
        ) if qoq_json else {}
        yoy_analysis = self.perform_comprehensive_analysis(
            current_json, current_chunks, yoy_json, yoy_chunks, "yoy",
            current_period=current_period_str,
            comparison_period=yoy_period_str
        ) if yoy_json else {}

        result = {
            "company": company,
            "current_period": f"{quarter} {year}",
            "previous_quarter": f"{prev_quarter} {prev_year}",
            "same_quarter_last_year": f"{quarter} {year-1}",
            "qoq_analysis": qoq_analysis,
            "yoy_analysis": yoy_analysis,
            "data_availability": {
                "current": True,
                "current_chunks": len(current_chunks),
                "previous_quarter": qoq_json is not None,
                "qoq_chunks": len(qoq_chunks),
                "same_quarter_last_year": yoy_json is not None,
                "yoy_chunks": len(yoy_chunks)
            }
        }

        # Save results
        save_success = self.save_comprehensive_analysis(company, quarter, year, result)
        result["save_status"] = "success" if save_success else "failed"

        return result

    def save_comprehensive_analysis(self, company: str, quarter: str, year: int, analysis_result: Dict) -> bool:
        """Save comprehensive analysis with Weaviate context to the processed intelligence JSON file."""
        filename = f"{company.lower()}_{quarter.lower()}_{year}.json"
        filepath = os.path.join("processed_intelligence", filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add comprehensive analysis with Weaviate context
            data["comprehensive_delta_analysis"] = {
                "qoq_analysis": analysis_result["qoq_analysis"],
                "yoy_analysis": analysis_result["yoy_analysis"],
                "comparison_periods": {
                    "current": analysis_result["current_period"],
                    "previous_quarter": analysis_result["previous_quarter"],
                    "same_quarter_last_year": analysis_result["same_quarter_last_year"]
                },
                "weaviate_context": {
                    "current_chunks_used": analysis_result["data_availability"]["current_chunks"],
                    "qoq_chunks_used": analysis_result["data_availability"]["qoq_chunks"],
                    "yoy_chunks_used": analysis_result["data_availability"]["yoy_chunks"]
                },
                "data_availability": {
                    "previous_quarter": analysis_result["data_availability"]["previous_quarter"],
                    "same_quarter_last_year": analysis_result["data_availability"]["same_quarter_last_year"]
                },
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_version": "comprehensive_v2_with_weaviate"
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Comprehensive analysis saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving analysis to {filepath}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Delta Analysis: Multi-Dimensional QoQ & YoY Comparative Analysis"
    )
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--quarter", required=True, help="Current quarter (Q1, Q2, Q3, Q4)")
    parser.add_argument("--year", type=int, required=True, help="Current year")
    parser.add_argument("--model", default="llama3.3:70b", help="LLM model to use (default: llama3.3:70b)")

    args = parser.parse_args()

    if args.quarter not in ["Q1", "Q2", "Q3", "Q4"]:
        print("Error: Quarter must be Q1, Q2, Q3, or Q4")
        return 1

    analyzer = ComprehensiveDeltaAnalyzer(model_name=args.model)
    result = analyzer.analyze_company(args.company, args.quarter, args.year)

    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE DELTA ANALYSIS (WITH WEAVIATE CONTEXT): {result['company'].upper()}")
    print(f"{'='*80}")
    print(f"Current Period: {result['current_period']}")
    print(f"Previous Quarter: {result['previous_quarter']}")
    print(f"Same Quarter Last Year: {result['same_quarter_last_year']}")
    print(f"\nWeaviate Context Retrieved:")
    print(f"  - Current Chunks: {result['data_availability']['current_chunks']}")
    print(f"  - QoQ Chunks: {result['data_availability']['qoq_chunks']}")
    print(f"  - YoY Chunks: {result['data_availability']['yoy_chunks']}")

    if result["data_availability"]["previous_quarter"] and result["qoq_analysis"]:
        print(f"\n{'='*80}")
        print("QUARTER-OVER-QUARTER (QoQ) ANALYSIS")
        print(f"{'='*80}")
        for key, value in result["qoq_analysis"].items():
            print(f"\n{key}")
            print(f"{'-'*80}")
            print(value[:1000] + "..." if len(value) > 1000 else value)

    if result["data_availability"]["same_quarter_last_year"] and result["yoy_analysis"]:
        print(f"\n{'='*80}")
        print("YEAR-OVER-YEAR (YoY) ANALYSIS")
        print(f"{'='*80}")
        for key, value in result["yoy_analysis"].items():
            print(f"\n{key}")
            print(f"{'-'*80}")
            print(value[:1000] + "..." if len(value) > 1000 else value)

    print(f"\n{'='*80}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
