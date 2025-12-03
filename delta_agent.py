"""
Delta Agent: QoQ & YoY Comparative Analysis

This script performs comparative analysis between quarters to identify strategic pivots,
tone shifts, and strategy changes in retail earnings transcripts.

Usage:
    python delta_agent.py --company "MondelezInternational" --ticker "MDLZ" --quarter "Q3" --year 2025
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from llm.ollama_client import OllamaClient

load_dotenv()

class DeltaAgent:
    """Agent for performing comparative analysis between quarters."""

    def __init__(self):
        self.client = self._init_weaviate()
        self.llm_client = OllamaClient()

    def _init_weaviate(self) -> weaviate.Client:
        """Initialize Weaviate client."""
        url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        api_key = os.getenv("WEAVIATE_API_KEY")
        auth = AuthApiKey(api_key) if api_key else None
        return weaviate.Client(url, auth_client_secret=auth, timeout_config=(5, 60))

    def get_transcript_data(self, company: str, ticker: str, quarter: str, year: int) -> str:
        """Fetch and aggregate transcript text for given parameters."""
        try:
            response = self.client.query.get(
                "RetailTranscriptChunk",
                ["text", "page", "chunk_index"]
            ).with_where({
                "operator": "And",
                "operands": [
                    {"path": ["company"], "operator": "Equal", "valueText": company},
                    {"path": ["ticker"], "operator": "Equal", "valueText": ticker},
                    {"path": ["quarter"], "operator": "Equal", "valueText": quarter},
                    {"path": ["year"], "operator": "Equal", "valueInt": year}
                ]
            }).with_sort([{"path": ["page"], "order": "asc"}, {"path": ["chunk_index"], "order": "asc"}]).do()

            objects = response.get("data", {}).get("Get", {}).get("RetailTranscriptChunk", [])
            if not objects:
                return ""

            # Aggregate and sort text by page and chunk index
            texts = []
            for obj in objects:
                texts.append(obj.get("text", ""))

            return "\n\n".join(texts)

        except Exception as e:
            print(f"Error fetching transcript data: {e}")
            return ""

    def get_previous_quarter(self, quarter: str, year: int) -> tuple[str, int]:
        """Get previous quarter and year."""
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        current_idx = quarters.index(quarter)

        if current_idx == 0:  # Q1 -> Q4 of previous year
            return "Q4", year - 1
        else:
            return quarters[current_idx - 1], year

    def compare_quarters(self, current_json: Dict, previous_json: Dict, same_quarter_json: Dict) -> str:
        """Perform comparative analysis using LLM."""

        prompt = f"""
You are a strategic analyst comparing retail earnings transcripts to identify key business pivots and shifts.

Compare the strategic focus between these periods and provide a concise summary.

**Current Period ({current_json['quarter']} {current_json['year']}):**
{current_json['text'][:4000]}...

**Previous Quarter ({previous_json['quarter']} {previous_json['year']}):**
{previous_json['text'][:4000]}...

**Same Quarter Last Year ({same_quarter_json['quarter']} {same_quarter_json['year']}):**
{same_quarter_json['text'][:4000]}...

**Analysis Requirements:**

1. **Strategic Pivots**: What topics did they STOP talking about compared to last year? What NEW topics emerged?

2. **Tone Shifts**: For key themes (Supply Chain, Inflation, Volume Growth, DTC, Retail Partners, etc.), is sentiment more positive/negative than last year?

3. **Strategy Changes**: Have they shifted focus between channels (DTC vs Retail Partners), product categories, or geographic markets?

**Output Format:**
Provide a concise summary (200-300 words) covering:
- Key pivots identified
- Notable tone shifts
- Strategic direction changes
- Business implications

Focus on actionable insights for investors and competitors.
"""

        try:
            response = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.strip()
        except Exception as e:
            return f"Error generating comparative analysis: {e}"

    def analyze_company(self, company: str, ticker: str, quarter: str, year: int) -> Dict:
        """Main analysis function."""

        # Get current transcript
        current_text = self.get_transcript_data(company, ticker, quarter, year)
        if not current_text:
            return {"error": f"No data found for {company} {ticker} {quarter} {year}"}

        # Get previous quarter
        prev_quarter, prev_year = self.get_previous_quarter(quarter, year)
        previous_text = self.get_transcript_data(company, ticker, prev_quarter, prev_year)

        # Get same quarter last year
        same_quarter_text = self.get_transcript_data(company, ticker, quarter, year - 1)

        # Prepare data for comparison
        current_data = {
            "quarter": quarter,
            "year": year,
            "text": current_text
        }

        previous_data = {
            "quarter": prev_quarter,
            "year": prev_year,
            "text": previous_text
        }

        same_quarter_data = {
            "quarter": quarter,
            "year": year - 1,
            "text": same_quarter_text
        }

        # Generate comparative analysis
        strategic_pivot_summary = self.compare_quarters(current_data, previous_data, same_quarter_data)

        return {
            "company": company,
            "ticker": ticker,
            "current_period": f"{quarter} {year}",
            "previous_quarter": f"{prev_quarter} {prev_year}",
            "same_quarter_last_year": f"{quarter} {year-1}",
            "strategic_pivot_summary": strategic_pivot_summary,
            "data_availability": {
                "current": len(current_text) > 0,
                "previous_quarter": len(previous_text) > 0,
                "same_quarter_last_year": len(same_quarter_text) > 0
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Delta Agent: QoQ & YoY Comparative Analysis")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--quarter", required=True, help="Current quarter (Q1, Q2, Q3, Q4)")
    parser.add_argument("--year", type=int, required=True, help="Current year")

    args = parser.parse_args()

    # Validate quarter
    if args.quarter not in ["Q1", "Q2", "Q3", "Q4"]:
        print("Error: Quarter must be Q1, Q2, Q3, or Q4")
        return 1

    # Run analysis
    agent = DeltaAgent()
    result = agent.analyze_company(args.company, args.ticker, args.quarter, args.year)

    # Output results
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    # Print summary
    print(f"\n=== Delta Agent Analysis: {result['company']} ({result['ticker']}) ===")
    print(f"Current Period: {result['current_period']}")
    print(f"Previous Quarter: {result['previous_quarter']}")
    print(f"Same Quarter Last Year: {result['same_quarter_last_year']}")
    print(f"\nData Availability: {result['data_availability']}")
    print(f"\n=== Strategic Pivot Summary ===")
    print(result['strategic_pivot_summary'])

    return 0


if __name__ == "__main__":
    sys.exit(main())