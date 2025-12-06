#!/usr/bin/env python3
"""
PDF Downloader Script

Downloads PDFs from firmographics table where is_downloaded is false,
renames them to company_quarter_year.pdf format, and saves to data/transcripts/.
Updates is_downloaded to true after successful download.
"""

import os
import requests
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = "postgresql+psycopg2://postgres:JustWin12@172.16.14.16:5432/crm_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def download_pdf(url: str, filepath: str) -> bool:
    """Download PDF from URL to filepath"""
    try:
        logger.info(f"📥 Downloading: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            f.write(response.content)

        logger.info(f"✅ Downloaded to: {filepath}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False

def get_company_name(session, company_id: str) -> str:
    """Get company name from companies table"""
    result = session.execute(
        text("SELECT name FROM companies WHERE id = :company_id"),
        {"company_id": company_id}
    ).fetchone()
    return result[0] if result else None

def parse_period(period: str) -> tuple:
    """Parse period like 'q3 2025' to ('Q3', '2025')"""
    if not period:
        return None, None

    parts = period.lower().split()
    if len(parts) == 2:
        quarter = parts[0].upper()  # q3 -> Q3
        year = parts[1]
        return quarter, year

    return None, None

def main():
    """Main download workflow"""
    transcripts_dir = Path("data/transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    session = SessionLocal()

    try:
        # Get all records where is_downloaded is false or NULL
        query = text("""
            SELECT id, company_id, last_quarterly_statement_url,
                   last_quarterly_statement_period
            FROM firmographics
            WHERE (is_downloaded = false OR is_downloaded IS NULL)
            AND last_quarterly_statement_url IS NOT NULL
        """)

        results = session.execute(query).fetchall()

        if not results:
            logger.info("ℹ️ No new PDFs to download")
            return

        logger.info(f"📋 Found {len(results)} PDFs to download")

        for row in results:
            record_id = row[0]
            company_id = row[1]
            url = row[2]
            period = row[3]

            # Get company name
            company_name = get_company_name(session, company_id)
            if not company_name:
                logger.error(f"❌ Company not found for ID: {company_id}")
                continue

            # Parse period
            quarter, year = parse_period(period)
            if not quarter or not year:
                logger.error(f"❌ Invalid period format: {period}")
                continue

            # Create filename: company_quarter_year.pdf
            filename = f"{company_name}_{quarter}_{year}.pdf"
            filepath = transcripts_dir / filename

            # Download PDF
            if download_pdf(url, str(filepath)):
                # Update is_downloaded to true
                update_query = text("""
                    UPDATE firmographics
                    SET is_downloaded = true, updated_at = NOW()
                    WHERE id = :record_id
                """)
                session.execute(update_query, {"record_id": record_id})
                session.commit()
                logger.info(f"✅ Updated database for {company_name}")
            else:
                logger.warning(f"⚠️ Skipping database update for {company_name}")

    except Exception as e:
        logger.error(f"❌ Error in download process: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    main()