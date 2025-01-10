from pathlib import Path

TEZZCHAIN_VERSION = "0.0.1"

TELEMETRY_API_KEY = "phc_4OU0g2ylf568KljSiiB7aiOyf6PPDMGspz5LmxMO5nG"
TELEMETRY_HOST = "https://us.i.posthog.com"

TEZZCHAIN_DIR = Path.home() / ".tezzchain"
TEZZCHAIN_DIR.mkdir(parents=True, exist_ok=True)

TEZZCHAIN_CONFIG_FILE = TEZZCHAIN_DIR / "config.json"

TEZZCHAIN_LOGGING_DIR = TEZZCHAIN_DIR / "logs"
TEZZCHAIN_LOGGING_DIR.mkdir(exist_ok=True)

TEZZCHAIN_DB = TEZZCHAIN_DIR / "chat.db"

TEZZCHAIN_TEMP_DIR = TEZZCHAIN_DIR / "temp"
TEZZCHAIN_TEMP_DIR.mkdir(exist_ok=True)
