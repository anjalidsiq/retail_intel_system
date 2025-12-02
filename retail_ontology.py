"""Retail Intelligence Ontology.

This module defines the core ontology for retail intelligence extraction,
including data schemas, validation rules, and domain concepts.
"""

import json
import logging
from typing import Dict, Any, List


# Collection name for Weaviate
COLLECTION_NAME = "RetailTranscriptChunk"

# System prompt to guide the LLM
SYSTEM_PROMPT = (
    "You are an expert Retail Intelligence Analyst with 15+ years of experience. "
    "Your task is to extract specific, actionable intelligence signals from earnings call transcripts.\n\n"
    "EXTRACTION RULES:\n"
    "1. PRIORITY: Executive Intent - Strategic Commitments:\n"
    "   - retail_media_budget: Specific investments, budget allocations, platform partnerships, ad revenue targets\n"
    "   - supply_chain_investment: Technology investments, efficiency programs, cost reduction commitments, automation\n"
    "   - digital_transformation_commitment: E-commerce growth targets, omnichannel investments, data platform builds\n"
    "   - premiumization_targets: Brand portfolio shifts, pricing architecture changes, SKU optimization goals\n"
    "   - strategic_priorities: CEO's key focus areas, investment commitments, growth initiatives (as array)\n\n"
    "2. Additional Intelligence Areas:\n"
    "   - retail_media_investments: Platforms, partnerships, ad revenue, RMN\n"
    "   - supply_chain_shelf_pain_points: OOS, inventory, logistics, margins\n"
    "   - category_strategy: Premiumization, DTC, omnichannel, SKU optimization\n"
    "   - financial_kpis: Revenue growth, margins, EPS, geographic breakdown\n"
    "   - strategic_initiatives: M&A, product launches, sustainability/ESG\n"
    "   - market_position: Market share, competitive positioning, brand equity\n"
    "   - digital_transformation: E-commerce growth, digital marketing ROI, CDP\n"
    "   - workforce_talent: Headcount, executive compensation, diversity\n"
    "   - future_outlook: 2025 guidance, long-term targets, capacity expansion\n"
    "   - risk_factors: Supply chain vulnerabilities, regulatory challenges, economic sensitivity\n"
    "   - partnerships_alliances: Retailer partnerships, supplier relationships, tech vendors\n"
    "   - capital_allocation: CapEx plans, dividend policy, share repurchase\n\n"
    "3. ONLY extract information that is EXPLICITLY mentioned in the provided text.\n"
    "4. If a field cannot be filled from the text, leave it as empty string/null.\n"
    "5. Do not infer, assume, or add external knowledge.\n"
    "6. Be precise - quote or paraphrase directly from the text.\n"
    "7. For leadership, extract names and titles only if mentioned.\n"
    "8. For firmographics, only extract numerical data if explicitly stated.\n"
    "9. CRITICAL: Return JSON that EXACTLY matches the provided schema structure. Do not create new fields or nested structures.\n"
    "10. Use the EXACT field names from the schema. Do not modify field names.\n\n"
    "Return only valid JSON in the exact schema provided. No explanations, no markdown."
)

# JSON schema template (the model must fill this based on the transcript)
JSON_SCHEMA_TEMPLATE = {
    "company_identity": {
        "brand_name": "",
        "parent_company": "",
        "ticker": "",
        "exchange": "",
        "website": ""
    },
    "firmographics": {
        "description": "",
        "industry_category": "",
        "revenue_ttm": 0,
        "currency": "USD",
        "employee_count": 0,
        "crm_segment": ""  # Will be calculated by script
    },
    "leadership": [],  # List of {"name": "", "title": ""}
    "intelligence": {
        "latest_transcript_url": "",
        "ir_landing_page": "",
        "executive_intent": {
            "retail_media_budget": "",
            "supply_chain_investment": "",
            "digital_transformation_commitment": "",
            "premiumization_targets": "",
            "strategic_priorities": []
        },
        "retail_media_investments": "",
        "supply_chain_shelf_pain_points": "",
        "category_strategy": "",
        "financial_kpis": {
            "revenue_growth": "",
            "margin_trends": "",
            "eps_performance": "",
            "geographic_breakdown": ""
        },
        "strategic_initiatives": {
            "ma_activity": "",
            "product_launches": "",
            "sustainability_esg": ""
        },
        "market_position": {
            "market_share_changes": "",
            "competitive_positioning": "",
            "brand_equity": ""
        },
        "digital_transformation": {
            "ecommerce_growth": "",
            "digital_marketing_roi": "",
            "customer_data_platform": ""
        },
        "workforce_talent": {
            "headcount_changes": "",
            "executive_compensation": "",
            "diversity_inclusion": ""
        },
        "future_outlook": {
            "guidance_2025": "",
            "long_term_targets": "",
            "capacity_expansion": ""
        },
        "risk_factors": {
            "supply_chain_vulnerabilities": "",
            "regulatory_challenges": "",
            "economic_sensitivity": ""
        },
        "partnerships_alliances": {
            "retailer_partnerships": "",
            "supplier_relationships": "",
            "technology_vendors": ""
        },
        "capital_allocation": {
            "capex_plans": "",
            "dividend_policy": "",
            "share_repurchase": ""
        }
    }
}

# Domain ontology anchors used for boosting / tagging
RETAIL_CONCEPTS: Dict[str, List[str]] = {
    "Retail_Media_AdTech": [
        "Retail Media Network",
        "RMN",
        "Amazon Advertising",
        "Walmart Connect",
        "Roundel",
        "First-Party Data",
        "Clean Room",
        "ROAS",
        "Ad Revenue",
    ],
    "Digital_Shelf_Friction": [
        "Out of Stock",
        "OOS",
        "Inventory Levels",
        "Supply Chain Constraints",
        "Fill Rate",
        "Margin Compression",
        "Promotional Intensity",
        "Unit Economics",
    ],
    "Channel_Strategy": [
        "DTC",
        "Direct to Consumer",
        "Omnichannel",
        "Click and Collect",
        "Brick and Mortar",
        "eCommerce Growth",
        "Marketplace",
    ],
    "Strategic_Levers": [
        "Price Pack Architecture",
        "Revenue Growth Management",
        "RGM",
        "Premiumization",
        "Cost Savings",
        "SKU Rationalization",
    ],
    "Financial_KPIs": [
        "Revenue Growth",
        "Top Line Growth",
        "Bottom Line",
        "Gross Margin",
        "Operating Margin",
        "Net Margin",
        "EPS Growth",
        "Cash Flow",
        "Free Cash Flow",
        "EBITDA",
        "Return on Investment",
        "ROI",
        "Market Share",
        "Profitability",
        "Cost of Goods Sold",
        "COGS",
    ],
    "Go_to_Market_Strategy": [
        "Distribution Network",
        "Channel Expansion",
        "Market Penetration",
        "Partnerships",
        "Strategic Alliances",
        "Mergers and Acquisitions",
        "M&A",
        "Joint Ventures",
        "Franchising",
        "Licensing",
        "Geographic Expansion",
        "Market Entry",
        "Competitive Positioning",
        "Brand Portfolio",
        "Product Portfolio",
    ],
    "Supply_Chain_Costs": [
        "Logistics Costs",
        "Transportation Costs",
        "Warehousing",
        "Procurement",
        "Supplier Relations",
        "Inventory Optimization",
        "Demand Forecasting",
        "Supply Chain Efficiency",
        "Cost Reduction",
        "Vendor Management",
        "Raw Material Costs",
        "Manufacturing Costs",
        "Distribution Costs",
        "Lead Times",
        "Supply Chain Resilience",
    ],
    "Digital_Channels": [
        "E-commerce",
        "Online Sales",
        "Mobile Commerce",
        "M-commerce",
        "Digital Transformation",
        "Data Analytics",
        "Customer Data Platform",
        "CDP",
        "Personalization",
        "Recommendation Engine",
        "Digital Marketing",
        "Social Commerce",
        "Marketplace Platforms",
        "App Development",
        "Website Optimization",
        "SEO",
        "SEM",
    ],
    "Operational_Levers": [
        "Cost Savings Initiatives",
        "Efficiency Improvements",
        "Process Optimization",
        "Automation",
        "Digital Automation",
        "Workforce Productivity",
        "Sustainability",
        "ESG",
        "Environmental Impact",
        "Carbon Footprint",
        "Sustainable Sourcing",
        "Energy Efficiency",
        "Waste Reduction",
        "Circular Economy",
        "Operational Excellence",
        "Lean Manufacturing",
        "Six Sigma",
    ],
}


def calculate_crm_segment(revenue_ttm: float) -> str:
    """Calculate CRM segment based on revenue."""
    try:
        if revenue_ttm >= 1_000_000_000:  # 1B USD
            return "Enterprise"
        else:
            return "SMB"
    except Exception as e:
        logging.error(f"Error calculating CRM segment: {e}")
        raise


def validate_extracted_data(data: Dict[str, Any]) -> bool:
    """Basic validation of the extracted JSON structure."""
    try:
        # More flexible validation - focus on having intelligence data
        if "intelligence" not in data:
            return False

        intelligence = data.get("intelligence", {})
        
        # Check that we have the core intelligence fields (even if empty)
        core_fields = ["retail_media_investments", "supply_chain_shelf_pain_points", "category_strategy"]
        has_core_structure = all(field in intelligence for field in core_fields)
        
        # Also check for executive_intent structure
        executive_intent = intelligence.get("executive_intent", {})
        has_executive_intent_structure = isinstance(executive_intent, dict) and all(key in executive_intent for key in [
            "retail_media_budget", "supply_chain_investment", "digital_transformation_commitment",
            "premiumization_targets", "strategic_priorities"
        ])

        # Accept data if it has proper structure, even if fields are empty
        if has_core_structure and has_executive_intent_structure:
            return True

        # Fallback: accept if we have any intelligence data at all
        return bool(intelligence)

    except Exception as e:
        logging.error(f"Error validating extracted data: {e}")
        return False


def generate_summary(data: Dict[str, Any]) -> str:
    """Generate a human-readable summary from extracted data."""
    try:
        summary_parts = []

        # Company info
        identity = data.get("company_identity", {})
        brand = identity.get("brand_name", "")
        parent = identity.get("parent_company", "")
        if brand:
            if parent and parent != brand:
                summary_parts.append(f"**{brand}** (part of {parent})")
            else:
                summary_parts.append(f"**{brand}**")

        # Intelligence highlights
        intel = data.get("intelligence", {})
        highlights = []

        # Executive Intent (highest priority)
        executive_intent = intel.get("executive_intent", {})
        if executive_intent.get("retail_media_budget"):
            highlights.append(f"💰 RM Budget: {executive_intent['retail_media_budget']}")
        if executive_intent.get("supply_chain_investment"):
            highlights.append(f"🏭 Supply Chain: {executive_intent['supply_chain_investment']}")
        if executive_intent.get("digital_transformation_commitment"):
            highlights.append(f"💻 Digital: {executive_intent['digital_transformation_commitment']}")
        if executive_intent.get("premiumization_targets"):
            highlights.append(f"⭐ Premium: {executive_intent['premiumization_targets']}")
        if executive_intent.get("strategic_priorities") and len(executive_intent["strategic_priorities"]) > 0:
            highlights.append(f"🎯 Priorities: {', '.join(executive_intent['strategic_priorities'])}")

        if intel.get("retail_media_investments"):
            highlights.append(f"📢 Retail Media: {intel['retail_media_investments']}")

        if intel.get("supply_chain_shelf_pain_points"):
            highlights.append(f"⚠️ Supply Chain: {intel['supply_chain_shelf_pain_points']}")

        if intel.get("category_strategy"):
            highlights.append(f"🎯 Strategy: {intel['category_strategy']}")

        # Add new intelligence categories
        financial_kpis = intel.get("financial_kpis", {})
        if financial_kpis.get("revenue_growth"):
            highlights.append(f"💰 Revenue Growth: {financial_kpis['revenue_growth']}")

        strategic = intel.get("strategic_initiatives", {})
        if strategic.get("sustainability_esg"):
            highlights.append(f"🌱 ESG: {strategic['sustainability_esg']}")

        digital = intel.get("digital_transformation", {})
        if digital.get("ecommerce_growth"):
            highlights.append(f"🛒 E-commerce: {digital['ecommerce_growth']}")

        future = intel.get("future_outlook", {})
        if future.get("guidance_2025"):
            highlights.append(f"🔮 2025 Outlook: {future['guidance_2025']}")

        if highlights:
            summary_parts.append("Key Intelligence:")
            summary_parts.extend(highlights)

        # Leadership
        leadership = data.get("leadership", [])
        if leadership:
            leaders = [f"{person.get('name', '')} ({person.get('title', '')})" for person in leadership if person.get('name')]
            if leaders:
                summary_parts.append("")  # Add empty line for separation
                summary_parts.append(f"Leadership: {', '.join(leaders)}")

        return "\n\n".join(summary_parts) if summary_parts else "No significant intelligence extracted."

    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return "Error generating summary."


def calculate_confidence_score(data: Dict[str, Any], source_chunks: int = 0) -> float:
    """Calculate a confidence score for the extracted data."""
    try:
        score = 0.0

        # Base score from data completeness
        intel = data.get("intelligence", {})
        intel_fields = ["retail_media_investments", "supply_chain_shelf_pain_points", "category_strategy"]
        filled_intel = 0
        for field in intel_fields:
            field_value = intel.get(field, "")
            if isinstance(field_value, dict):
                # For nested objects, check if any subfield has content
                if any(str(v).strip() for v in field_value.values() if v):
                    filled_intel += 1
            elif field_value and str(field_value).strip():
                filled_intel += 1

        if filled_intel >= 2:
            score += 0.25
        elif filled_intel == 1:
            score += 0.15

        # Executive Intent bonus (highest priority)
        executive_intent = intel.get("executive_intent", {})
        if isinstance(executive_intent, dict):
            filled_executive_fields = sum(1 for v in executive_intent.values() if v and str(v).strip())
            if filled_executive_fields > 0:
                score += min(filled_executive_fields * 0.1, 0.3)  # Up to 0.3 bonus for executive intent

        # New intelligence categories scoring
        new_categories = [
            "financial_kpis", "strategic_initiatives", "market_position",
            "digital_transformation", "workforce_talent", "future_outlook",
            "risk_factors", "partnerships_alliances", "capital_allocation"
        ]

        filled_new_categories = 0
        for category in new_categories:
            category_data = intel.get(category, {})
            if isinstance(category_data, dict):
                # Count filled fields in nested objects
                filled_fields = sum(1 for v in category_data.values() if v and str(v).strip())
                if filled_fields > 0:
                    filled_new_categories += 1
            elif category_data and str(category_data).strip():
                filled_new_categories += 1

        # Bonus for new categories (up to 0.25 points)
        score += min(filled_new_categories * 0.025, 0.25)

        # Company identity completeness
        identity = data.get("company_identity", {})
        if identity.get("brand_name"):
            score += 0.1

        # Leadership information
        if data.get("leadership"):
            score += 0.05

        # Source chunks factor
        if source_chunks > 10:
            score += 0.15
        elif source_chunks > 5:
            score += 0.1

        # Revenue data (indicates more complete firmographics)
        if data.get("firmographics", {}).get("revenue_ttm", 0) > 0:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    except Exception as e:
        logging.error(f"Error calculating confidence score: {e}")
        return 0.0


def normalize_executive_intent(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize executive_intent section to match expected schema."""
    try:
        intelligence = data.get("intelligence", {})

        # If executive_intent is already properly structured, return as-is
        executive_intent = intelligence.get("executive_intent", {})
        if isinstance(executive_intent, dict) and all(key in executive_intent for key in [
            "retail_media_budget", "supply_chain_investment", "digital_transformation_commitment",
            "premiumization_targets", "strategic_priorities"
        ]):
            return data

        # Create normalized executive_intent
        normalized_executive_intent = {
            "retail_media_budget": "",
            "supply_chain_investment": "",
            "digital_transformation_commitment": "",
            "premiumization_targets": "",
            "strategic_priorities": []
        }

        # Try to extract from various possible field names/locations
        possible_fields = {
            "retail_media_budget": ["Retail Media Budget", "retail_media_budget", "Retail Media Investments", "retail_media_budget"],
            "supply_chain_investment": ["Supply Chain Investment", "supply_chain_investment", "Supply Chain/Shelf Pain Points", "supply_chain_investment"],
            "digital_transformation_commitment": ["Digital Transformation Commitment", "digital_transformation_commitment", "Digital Transformation", "digital_transformation_commitment"],
            "premiumization_targets": ["Premiumization Targets", "premiumization_targets", "Category Strategy", "premiumization_targets", "shift to premium"],
            "strategic_priorities": ["Strategic Priorities", "strategic_priorities", "strategic_commitments"]
        }

        for target_field, source_fields in possible_fields.items():
            for source_field in source_fields:
                if source_field in intelligence:
                    value = intelligence[source_field]
                    if isinstance(value, dict):
                        # Convert dict to string summary
                        filled_parts = [f"{k}: {v}" for k, v in value.items() if v and str(v).strip()]
                        if filled_parts:
                            normalized_executive_intent[target_field] = "; ".join(filled_parts)
                    elif isinstance(value, list):
                        if target_field == "strategic_priorities":
                            normalized_executive_intent[target_field] = value
                        else:
                            normalized_executive_intent[target_field] = ", ".join(str(v) for v in value if v)
                    elif value and str(value).strip():
                        normalized_executive_intent[target_field] = str(value)
                    break  # Use first matching field

        # Update the data structure
        intelligence["executive_intent"] = normalized_executive_intent
        data["intelligence"] = intelligence

        return data

    except Exception as e:
        logging.error(f"Error normalizing executive_intent: {e}")
        return data


def comprehensive_normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive normalization to map LLM creative fields to exact schema."""
    try:
        # Start with base schema structure
        normalized = {
            "company_identity": {
                "brand_name": "",
                "parent_company": "",
                "ticker": "",
                "exchange": "",
                "website": ""
            },
            "firmographics": {
                "description": "",
                "industry_category": "",
                "revenue_ttm": 0,
                "currency": "USD",
                "employee_count": 0,
                "crm_segment": "SMB"
            },
            "leadership": [],
            "intelligence": JSON_SCHEMA_TEMPLATE["intelligence"].copy()
        }

        # Extract firmographics from various locations
        if "firmographics" in data:
            normalized["firmographics"].update(data["firmographics"])
        
        # Also check for revenue_ttm in intelligence section (LLM sometimes puts it there)
        if "intelligence" in data and "revenue_ttm" in data["intelligence"]:
            normalized["firmographics"]["revenue_ttm"] = data["intelligence"]["revenue_ttm"]
        
        # Also check for revenue_ttm at top level
        if "revenue_ttm" in data:
            normalized["firmographics"]["revenue_ttm"] = data["revenue_ttm"]

        # Extract leadership from various locations
        leadership_sources = ["leadership"]
        if "intelligence" in data and "leadership" in data["intelligence"]:
            leadership_sources.append("intelligence.leadership")

        for source in leadership_sources:
            if "." in source:
                section, field = source.split(".", 1)
                leaders = data.get(section, {}).get(field, [])
            else:
                leaders = data.get(source, [])

            if leaders and isinstance(leaders, list):
                normalized["leadership"] = leaders
                break

        # Normalize intelligence section
        intel = data.get("intelligence", {})

        # First, copy any existing executive_intent fields directly
        if "executive_intent" in intel and isinstance(intel["executive_intent"], dict):
            normalized["intelligence"]["executive_intent"].update(intel["executive_intent"])

        # Also copy any executive intent fields that might be at the top level of intelligence
        executive_intent_fields = [
            "retail_media_budget", "supply_chain_investment", "digital_transformation_commitment",
            "premiumization_targets", "strategic_priorities"
        ]
        for field in executive_intent_fields:
            if field in intel and intel[field]:
                normalized["intelligence"]["executive_intent"][field] = intel[field]

        # Map financial KPIs
        if "financial_performance" in intel:
            fp = intel["financial_performance"]
            if "revenue_growth" in fp:
                normalized["intelligence"]["financial_kpis"]["revenue_growth"] = fp["revenue_growth"]
            if "operating_margin" in fp:
                normalized["intelligence"]["financial_kpis"]["margin_trends"] = fp["operating_margin"]

        # Map quarterly performance
        if "quarterly_performance" in data:
            qp = data["quarterly_performance"]
            if "revenue_growth" in qp:
                normalized["intelligence"]["financial_kpis"]["revenue_growth"] = qp["revenue_growth"]

        # Map product categories to category strategy
        if "product_categories" in intel:
            categories = intel["product_categories"]
            if isinstance(categories, list):
                normalized["intelligence"]["category_strategy"] = ", ".join(categories)

        # Extract strategic priorities from various locations
        strategic_sources = ["strategic_priorities", "strategic_commitments"]
        for source in strategic_sources:
            if source in intel:
                priorities = intel[source]
                if isinstance(priorities, list) and priorities:
                    normalized["intelligence"]["executive_intent"]["strategic_priorities"] = priorities
                    break

        # If strategic_priorities is already in executive_intent, keep it
        if "executive_intent" in intel and "strategic_priorities" in intel["executive_intent"]:
            existing_priorities = intel["executive_intent"]["strategic_priorities"]
            if existing_priorities:
                normalized["intelligence"]["executive_intent"]["strategic_priorities"] = existing_priorities

        # Copy other intelligence fields directly
        other_intel_fields = [
            "retail_media_investments", "supply_chain_shelf_pain_points", "category_strategy",
            "financial_kpis", "strategic_initiatives", "market_position", "digital_transformation",
            "workforce_talent", "future_outlook", "risk_factors", "partnerships_alliances", "capital_allocation"
        ]
        for field in other_intel_fields:
            if field in intel and intel[field]:
                if isinstance(normalized["intelligence"][field], dict) and isinstance(intel[field], dict):
                    normalized["intelligence"][field].update(intel[field])
                else:
                    normalized["intelligence"][field] = intel[field]

        # Normalize executive intent (this will handle any existing executive_intent section)
        normalized = normalize_executive_intent(normalized)

        return normalized

    except Exception as e:
        logging.error(f"Error in comprehensive normalization: {e}")
        return data