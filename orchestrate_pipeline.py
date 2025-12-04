#!/usr/bin/env python3
"""
Main Orchestrator: Automates the complete retail intelligence pipeline

Watches `data/transcripts/` for new PDFs named like Company_Q3_2025.pdf
and runs ingestion, extraction, and delta comparison.
"""

import subprocess
import os
import json
import sys
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFWatcherHandler(FileSystemEventHandler):
	"""Watch for new PDF files and trigger pipeline"""

	def on_created(self, event):
		# ignore directories
		if event.is_directory:
			return

		# only handle pdf files
		if not event.src_path.lower().endswith('.pdf'):
			return

		logger.info(f"📄 New PDF detected: {event.src_path}")

		try:
			filename = os.path.basename(event.src_path)
			company, quarter, year = self.extract_metadata(filename)
			# Run pipeline in background so watcher keeps running
			subprocess.Popen([
				sys.executable, os.path.join(os.getcwd(), "orchestrate_pipeline.py"),
				"--run", 
				"--company", company,
				"--quarter", quarter,
				"--year", str(year)
			])

			logger.info(f"✅ Pipeline triggered for {company} {quarter} {year}")

		except ValueError as e:
			logger.error(f"❌ Failed to parse filename: {e}")
		except Exception as e:
			logger.error(f"❌ Error triggering pipeline: {e}")

	def extract_metadata(self, filename):
		"""Extract company, quarter, year from filename
		Expected format: Company_Q3_2025.pdf or Company-Q3-2025.pdf
		"""
		name = filename.rsplit('.', 1)[0]
		# accept _ or - separators
		for sep in ('_', '-'):
			parts = name.split(sep)
			if len(parts) >= 3:
				company = parts[0]
				quarter = parts[1]
				try:
					year = int(parts[2])
				except ValueError:
					raise ValueError(f"Invalid year in filename: {parts[2]}")

				if quarter not in ("Q1", "Q2", "Q3", "Q4"):
					raise ValueError(f"Invalid quarter in filename: {quarter}")

				return company, quarter, year

		raise ValueError("Filename must be Company_Q3_2025.pdf or Company-Q3-2025.pdf")


def run_pipeline(company: str, quarter: str, year: int):
	"""Execute complete 3-task pipeline (ingest -> extract -> delta)"""

	logger.info(f"\n{'='*60}")
	logger.info(f"🚀 Starting Pipeline: {company} {quarter} {year}")
	logger.info(f"{'='*60}\n")

	try:
		# TASK 3.1: INGESTION
		logger.info("📥 TASK 3.1: Ingesting transcripts...")
		result = subprocess.run([
			sys.executable, os.path.join(os.getcwd(), "ingestion/ingest_transcripts.py"),
			"--call-type", "earnings_call",
			"--delete-existing"
		], capture_output=True, text=True, timeout=600)

		if result.returncode != 0:
			logger.error(f"❌ Ingestion Failed: {result.stderr}")
			return 1
		logger.info("✅ Ingestion Complete\n")

		# TASK 3.2: EXTRACTION
		logger.info("🔍 TASK 3.2: Extracting intelligence...")
		result = subprocess.run([
			sys.executable, os.path.join(os.getcwd(), "output/extract_quarterly_intel.py"),
			"--company", company,
			"--quarter", quarter,
			"--year", str(year),
			"--verbose"
		], capture_output=True, text=True, timeout=900)

		if result.returncode != 0:
			logger.error(f"❌ Extraction Failed: {result.stderr}")
			return 1
		logger.info("✅ Extraction Complete\n")

		# TASK 3.3: DELTA AGENT
		logger.info("⚖️ TASK 3.3: Running Delta Agent...")
		result = subprocess.run([
			sys.executable, os.path.join(os.getcwd(), "delta_agent.py"),
			"--company", company,
			"--quarter", quarter,
			"--year", str(year)
		], capture_output=True, text=True, timeout=300)

		if result.returncode != 0:
			# Delta agent may fail when previous data missing — warn but continue
			logger.warning(f"⚠️ Delta Agent finished with warning/error: {result.stderr}")
		else:
			logger.info("✅ Delta Agent Complete\n")

		logger.info(f"🎉 Pipeline completed successfully for {company} {quarter} {year}")
		return 0

	except subprocess.TimeoutExpired:
		logger.error("❌ Pipeline timed out")
		return 1
	except Exception as e:
		logger.error(f"❌ Pipeline failed with exception: {e}")
		return 1


def start_watcher():
	"""Start the file watcher to monitor for new PDFs"""
	watch_path = Path("data/transcripts")
	watch_path.mkdir(parents=True, exist_ok=True)

	logger.info(f"👀 Starting PDF watcher on: {watch_path.absolute()}")
	logger.info("📄 Expected filename format: Company_Q3_2025.pdf")
	logger.info("🛑 Press Ctrl+C to stop\n")

	event_handler = PDFWatcherHandler()
	observer = Observer()
	observer.schedule(event_handler, str(watch_path), recursive=False)
	observer.start()

	try:
		observer.join()
	except KeyboardInterrupt:
		logger.info("\n🛑 Stopping watcher...")
		observer.stop()
		observer.join()


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Retail Intelligence Pipeline Orchestrator")
	parser.add_argument(
		"--run",
		action="store_true",
		help="Run the pipeline for specific parameters"
	)
	parser.add_argument(
		"--company",
		help="Company name"
	)
	parser.add_argument(
		"--quarter",
		help="Quarter (Q1, Q2, Q3, Q4)"
	)
	parser.add_argument(
		"--year",
		type=int,
		help="Year"
	)
	parser.add_argument(
		"--watch",
		action="store_true",
		help="Start file watcher mode (default)"
	)

	args = parser.parse_args()

	if args.run:
		# Validate required args for run mode
		if not all([args.company, args.quarter, args.year]):
			parser.error("--run requires --company, --quarter, and --year")
		
		sys.exit(run_pipeline(args.company, args.quarter, args.year))
	else:
		# Default to watcher mode
		start_watcher()


if __name__ == "__main__":
	main()
