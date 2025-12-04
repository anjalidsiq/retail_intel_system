"""
Delta Agent: QoQ & YoY Comparative Analysis

This script performs comparative analysis between quarters to identify strategic pivots,
tone shifts, and strategy changes using processed intelligence outputs.

Usage:
    python delta_agent.py --company "Unilever" --quarter "Q3" --year 2025
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from llm.ollama_client import OllamaClient

load_dotenv()

class DeltaAgent:
    """Agent for performing comparative analysis between quarters using processed intelligence."""

    def __init__(self):
        self.llm_client = OllamaClient()

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
            print(f"Error loading {filepath}: {e}")
            return None

    def get_previous_quarter(self, quarter: str, year: int) -> tuple[str, int]:
        """Get previous quarter and year."""
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        current_idx = quarters.index(quarter)

        if current_idx == 0:  # Q1 -> Q4 of previous year
            return "Q4", year - 1
        else:
            return quarters[current_idx - 1], year

    def compare_quarters(self, current_json: Dict, previous_json: Optional[Dict], same_quarter_json: Optional[Dict]) -> str:
        """Perform comparative analysis using LLM on processed intelligence outputs."""

        # Extract key sections from current period
        current_strategic = current_json.get("account_data_map", {}).get("strategic_priority_summary", "")
        current_risks = current_json.get("account_data_map", {}).get("risk_summary", "")
        current_leadership = ", ".join([f"{p['full_name']} ({p['job_title']})" for p in current_json.get("contact_data_map", [])])
        
        # Extract from previous quarter (if available)
        prev_strategic = previous_json.get("account_data_map", {}).get("strategic_priority_summary", "Not available") if previous_json else "Not available"
        prev_risks = previous_json.get("account_data_map", {}).get("risk_summary", "Not available") if previous_json else "Not available"
        
        # Extract from same quarter last year (if available)
        yoy_strategic = same_quarter_json.get("account_data_map", {}).get("strategic_priority_summary", "Not available") if same_quarter_json else "Not available"
        yoy_risks = same_quarter_json.get("account_data_map", {}).get("risk_summary", "Not available") if same_quarter_json else "Not available"

        yoy_period = f"{same_quarter_json['_metadata']['quarter']} {same_quarter_json['_metadata']['year']}" if same_quarter_json else "previous year data not available"
        yoy_header = f"{same_quarter_json['_metadata']['quarter']} {same_quarter_json['_metadata']['year']}" if same_quarter_json else "N/A"

        prompt = f"""
Compare the strategic focus of {current_json['_metadata']['quarter']} {current_json['_metadata']['year']} vs {yoy_period}.

**Current Period ({current_json['_metadata']['quarter']} {current_json['_metadata']['year']}):**
Strategic Priorities: {current_strategic}
Risk Summary: {current_risks}
Leadership: {current_leadership}

**Previous Quarter ({previous_json['_metadata']['quarter']} {previous_json['_metadata']['year']} if previous_json else 'N/A'):**
Strategic Priorities: {prev_strategic}
Risk Summary: {prev_risks}

**Same Quarter Last Year ({yoy_header}):**
Strategic Priorities: {yoy_strategic}
Risk Summary: {yoy_risks}

**Analysis Requirements:**

Identify Pivots: What did they stop talking about? (e.g., Stopped mentioning 'Inflation', started mentioning 'Volume Growth').

Tone Shift: Is the sentiment regarding 'Supply Chain' more positive or negative than last year?

Strategy Change: Have they shifted focus from 'DTC' to 'Retail Partners'?

**Output:** A simple text summary field called strategic_pivot_summary with clear headings for each analysis area (200-300 words total).

**Identify Pivots:**
[Analysis of what they stopped talking about vs. new topics]

**Tone Shift:**
[Analysis of sentiment changes regarding Supply Chain]

**Strategy Change:**
[Analysis of shifts from DTC to Retail Partners]
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

    def analyze_company(self, company: str, quarter: str, year: int) -> Dict:
        """Main analysis function using processed intelligence JSONs."""

        # Load current period intelligence
        current_json = self.load_processed_intelligence(company, quarter, year)
        if not current_json:
            return {"error": f"No processed intelligence found for {company} {quarter} {year}"}

        # Load previous quarter
        prev_quarter, prev_year = self.get_previous_quarter(quarter, year)
        previous_json = self.load_processed_intelligence(company, prev_quarter, prev_year)

        # Load same quarter last year
        same_quarter_json = self.load_processed_intelligence(company, quarter, year - 1)

        # Generate comparative analysis
        strategic_pivot_summary = self.compare_quarters(current_json, previous_json, same_quarter_json)

        return {
            "company": company,
            "current_period": f"{quarter} {year}",
            "previous_quarter": f"{prev_quarter} {prev_year}",
            "same_quarter_last_year": f"{quarter} {year-1}",
            "strategic_pivot_summary": strategic_pivot_summary,
            "data_availability": {
                "current": True,
                "previous_quarter": previous_json is not None,
                "same_quarter_last_year": same_quarter_json is not None
            }
        }




def main():
    parser = argparse.ArgumentParser(description="Delta Agent: QoQ & YoY Comparative Analysis")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--quarter", required=True, help="Current quarter (Q1, Q2, Q3, Q4)")
    parser.add_argument("--year", type=int, required=True, help="Current year")

    args = parser.parse_args()

    # Validate quarter
    if args.quarter not in ["Q1", "Q2", "Q3", "Q4"]:
        print("Error: Quarter must be Q1, Q2, Q3, or Q4")
        return 1

    # Run analysis
    agent = DeltaAgent()
    result = agent.analyze_company(args.company, args.quarter, args.year)

    # Output results
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    # Print summary
    print(f"\n=== Delta Agent Analysis: {result['company']} ===")
    print(f"Current Period: {result['current_period']}")
    print(f"Previous Quarter: {result['previous_quarter']}")
    print(f"Same Quarter Last Year: {result['same_quarter_last_year']}")
    print(f"\nData Availability: {result['data_availability']}")
    print(f"\n=== Strategic Pivot Summary ===")
    print(result['strategic_pivot_summary'])

    return 0


if __name__ == "__main__":
    sys.exit(main())