import os
import hvac
from dotenv import load_dotenv

load_dotenv(override=True)  

def load_vault_secrets():
    """
    Connects to Vault and loads secrets into environment variables.
    """
    vault_addr = os.getenv("VAULT_ADDR", "https://vault.media-amp.com")
    client = hvac.Client(url=vault_addr, verify=True)

    # For local development (LDAP)
    if os.getenv("VAULT_USER") and os.getenv("VAULT_PASS"):
        client.auth.ldap.login(
            username=os.getenv("VAULT_USER"),
            password=os.getenv("VAULT_PASS")
        )

    # For production (AppRole)
    elif os.getenv("VAULT_ROLE_ID") and os.getenv("VAULT_SECRET_ID"):
        client.auth.approle.login(
            role_id=os.getenv("VAULT_ROLE_ID"),
            secret_id=os.getenv("VAULT_SECRET_ID")
        )

    else:
        raise RuntimeError("Vault credentials not found in environment.")

    # Read the app’s configuration secret
    secret = client.secrets.kv.v2.read_secret_version(
        path="retail_intel_system", mount_point="mediaamp/dev"
    )

    data = secret["data"]["data"]

    for key, value in data.items():
        os.environ[key] = str(value)

    print("✅ Vault secrets loaded into environment")