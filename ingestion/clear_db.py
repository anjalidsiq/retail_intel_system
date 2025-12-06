import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import vault_client
sys.path.insert(0, str(Path(__file__).parent.parent))
from vault_client import load_vault_secrets
import weaviate
from weaviate.auth import AuthApiKey

load_vault_secrets()

# Connect to Weaviate
url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
api_key = os.getenv("WEAVIATE_API_KEY")
auth = AuthApiKey(api_key) if api_key else None

client = weaviate.Client(url, auth_client_secret=auth, timeout_config=(5, 60))

# Delete the collection
try:
    client.schema.delete_class("RetailTranscriptChunk")
    print("✓ RetailTranscriptChunk collection deleted successfully")
except Exception as e:
    print(f"✗ Error deleting collection: {e}")
