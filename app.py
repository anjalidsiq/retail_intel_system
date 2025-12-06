"""
Retail Intelligence System - Main Application
A Flask-based web application providing endpoints for retail transcript processing.
"""

import os
import subprocess
import sys
from flask import Flask, request, jsonify
from vault_client import load_vault_secrets

# Load Vault secrets on startup
load_vault_secrets()

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "Retail Intelligence System API",
        "endpoints": {
            "/download": "POST - Download PDFs from database",
            "/ingest": "POST - Ingest transcripts (params: input_dir, use_local_embeddings, delete_existing)",
            "/extract": "POST - Extract intelligence (params: company, quarter, year)",
            "/delta": "POST - Run delta analysis (params: company, quarter, year)",
            "/health": "GET - Health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "vault_loaded": True})

@app.route('/download', methods=['POST'])
def download():
    """Download PDFs from database."""
    cmd = [
        sys.executable, 'data/scripts/download_pdfs.py'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return jsonify({
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/ingest', methods=['POST'])
def ingest():
    """Trigger transcript ingestion."""
    data = request.get_json() or {}
    input_dir = data.get('input_dir', 'data/transcripts')
    use_local_embeddings = data.get('use_local_embeddings', False)
    delete_existing = data.get('delete_existing', False)

    cmd = [sys.executable, 'ingestion/ingest_transcripts.py']
    if use_local_embeddings:
        cmd.append('--use-local-embeddings')
    if delete_existing:
        cmd.append('--delete-existing')
    cmd.extend(['--input-dir', input_dir])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return jsonify({
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract():
    """Extract quarterly intelligence."""
    data = request.get_json() or {}
    company = data.get('company')
    quarter = data.get('quarter')
    year = data.get('year')

    if not all([company, quarter, year]):
        return jsonify({"error": "Missing required params: company, quarter, year"}), 400

    cmd = [
        sys.executable, 'output/extract_quarterly_intel.py',
        '--company', company,
        '--quarter', quarter,
        '--year', str(year)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return jsonify({
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/delta', methods=['POST'])
def delta():
    """Run delta analysis."""
    data = request.get_json() or {}
    company = data.get('company')
    quarter = data.get('quarter')
    year = data.get('year')

    if not all([company, quarter, year]):
        return jsonify({"error": "Missing required params: company, quarter, year"}), 400

    cmd = [
        sys.executable, 'delta_agent.py',
        '--company', company,
        '--quarter', quarter,
        '--year', str(year)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return jsonify({
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)