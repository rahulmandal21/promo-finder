"""All tunables in one place. Provider/model are chosen ONLY via env vars."""
import os
from pathlib import Path


def _load_dotenv():
    """Tiny .env reader (KEY=VALUE lines). Real environment variables always win."""
    f = Path(__file__).with_name(".env")
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# ---- clip rules (from the brief) ----
MIN_TOTAL, MAX_TOTAL = 20.0, 60.0
HOOK_MIN, HOOK_MAX = 3.0, 12.0
BRIDGE_MIN, BRIDGE_MAX = 5.0, 20.0
TWIST_MIN, TWIST_MAX = 10.0, 25.0
MAX_SEGMENTS = 4
HOOK_WINDOW = 3.0          # a hook must land inside the first 3 s
TOL = 0.05                 # float tolerance when comparing timestamps
CANDS_PER_EPISODE = (12, 16)

# ---- ranking ----
WEIGHTS = {"hook": 0.35, "cliffhanger": 0.35, "standalone": 0.30}
PENALTIES = {              # subtracted from the 0-10 weighted score
    "unverified": 2.0,     # quoted hook/cliffhanger not found at claimed place
    "quiet_start": 1.0,    # opens on greeting / no spoken or shocking beat in 3 s
    "cue_only": 1.5,       # cliffhanger rests only on writers' SOUND FX cue
    "needs_earlier": 1.0,  # blind viewer could not follow it cold
    "hook_late": 0.5,      # verified hook line lands after 3.5 s
}
BLIND_NO_PENALTY = 3.0     # standalone score drop when blind viewer says "no"
BLIND_TOP_K = int(os.getenv("PROMO_BLIND_K", "5"))   # clips per episode sent to blind viewer
KEEP_PER_EPISODE = 3
MIN_EXTRA_SCORE = 6.0      # 2nd/3rd pick per episode must clear this
DUP_OVERLAP = 0.3          # overlap share that counts as near-duplicate
POST_FIRST_N = 5
SKIP_MIN_BASE = 7.0        # "high-scoring" threshold for the skipped-for-context list
MONOCULTURE = 0.70         # flag if >70% of top picks share one shape
HIT_OVERLAP = 0.5          # evaluate.py: overlap share that counts as a hit

# ---- LLM plumbing ----
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
    "mock": "mock",
}
TIMEOUT = 180
MAX_429_RETRIES = 6
MAX_5XX_RETRIES = 5
WAIT_429 = float(os.getenv("PROMO_429_WAIT", "65"))     # seconds; env override is for tests
BACKOFF_BASE = float(os.getenv("PROMO_BACKOFF", "5"))


def provider(suffix=""):
    return os.getenv("PROMO_PROVIDER" + suffix, "gemini" if not suffix else "").strip().lower()


def model(prov, suffix=""):
    return os.getenv("PROMO_MODEL" + suffix, "").strip() or DEFAULT_MODELS.get(prov, "")


def delay():
    return float(os.getenv("PROMO_DELAY", "8"))


KEY_ENV = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}


def api_key(prov):
    for k in KEY_ENV.get(prov, []):
        if os.getenv(k):
            return os.getenv(k)
    return None
