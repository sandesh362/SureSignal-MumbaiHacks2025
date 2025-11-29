# backend/config.py
import os
import logging
from typing import Dict, List, Any

# Try to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)


class Config:
    # Flask / environment
    FLASK_ENV: str = os.getenv("FLASK_ENV", "production")
    DEBUG: bool = FLASK_ENV == "development"

    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "veripulse")

    # Social API credentials (optional for simulation)
    TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    TWITTER_API_KEY: str = os.getenv("TWITTER_API_KEY", "")
    TWITTER_API_SECRET: str = os.getenv("TWITTER_API_SECRET", "")
    TWITTER_ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN", "")
    TWITTER_ACCESS_SECRET: str = os.getenv("TWITTER_ACCESS_SECRET", "")

    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "veripulse-bot/0.1")

    # Vector store / embeddings
    VECTOR_PROVIDER: str = os.getenv("VECTOR_PROVIDER", "chromadb")  # or "inmemory" or "pinecone"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", None)

    # LLM
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")
    # Note: API keys (OpenAI, Google) should be in environment as normal (OPENAI_API_KEY, etc.)

    # Application behavior / thresholds
    MAX_SOURCES_PER_CLAIM: int = int(os.getenv("MAX_SOURCES_PER_CLAIM", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # e.g. per hour

    # Trusted sources & RSS feeds (defaults - add more in .env if you want)
    TRUSTED_SOURCES: Dict[str, str] = {
        "pib": "https://pib.gov.in",
        "who": "https://www.who.int",
        "imd": "https://mausam.imd.gov.in",
        "ndma": "https://ndma.gov.in",
        "reuters": "https://www.reuters.com",
        "ap_news": "https://apnews.com",
    }

    RSS_FEEDS: List[str] = [
        os.getenv("RSS_FEED_1", "https://www.reutersagency.com/feed/?best-topics=world"),
        os.getenv("RSS_FEED_2", "https://www.who.int/feeds/entity/csr/don/en/rss.xml"),
    ]

    # Optional feature flags
    ENABLE_MULTIMODAL: bool = os.getenv("ENABLE_MULTIMODAL", "false").lower() in ("1", "true", "yes")

    @classmethod
    def validate(cls) -> None:
        """
        Validate essential configuration and warn about optional missing values.
        Raises ValueError on critical missing config.
        """
        missing = []
        critical = []

        # Critical: MongoDB is required for this project (our services expect it)
        if not cls.MONGODB_URI:
            critical.append("MONGODB_URI")

        # Warning if both social creds are missing (we will run in simulation then)
        twitter_creds = any([
            cls.TWITTER_BEARER_TOKEN,
            cls.TWITTER_API_KEY and cls.TWITTER_API_SECRET and cls.TWITTER_ACCESS_TOKEN and cls.TWITTER_ACCESS_SECRET
        ])
        reddit_creds = bool(cls.REDDIT_CLIENT_ID and cls.REDDIT_CLIENT_SECRET)

        # LLM note: we don't require the key here, but warn if DEFAULT_LLM_MODEL is set to a hosted model
        if "gpt" in cls.DEFAULT_LLM_MODEL.lower():
            if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_BASE"):
                logger.warning("DEFAULT_LLM_MODEL refers to an OpenAI model but OPENAI_API_KEY/OPENAI_API_BASE is not set.")

        # Vector provider check
        if cls.VECTOR_PROVIDER.lower() == "chromadb":
            # chroma dir only required if you expect persistence
            if cls.CHROMA_PERSIST_DIR is None:
                logger.info("CHROMA_PERSIST_DIR not set — Chromadb will run in default (temp) mode if available.")

        if critical:
            raise ValueError(f"Missing critical config values: {', '.join(critical)}. "
                             "Set them in your environment or .env file and restart.")

        # Informational logs
        if not twitter_creds and not reddit_creds:
            logger.info("No social media credentials found — ingestion agent will run in simulation mode.")
        else:
            if not twitter_creds:
                logger.info("Twitter credentials incomplete or missing — Twitter ingestion disabled.")
            if not reddit_creds:
                logger.info("Reddit credentials incomplete or missing — Reddit ingestion disabled.")

        # Validate numeric thresholds
        if cls.SIMILARITY_THRESHOLD <= 0 or cls.SIMILARITY_THRESHOLD > 1.0:
            logger.warning("SIMILARITY_THRESHOLD should be between 0 and 1. Using default 0.7.")
            cls.SIMILARITY_THRESHOLD = 0.7

        if cls.MAX_SOURCES_PER_CLAIM <= 0:
            logger.warning("MAX_SOURCES_PER_CLAIM must be positive. Using default 5.")
            cls.MAX_SOURCES_PER_CLAIM = 5

        logger.info("Config validated successfully.")

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return config subset useful for logging / debugging."""
        return {
            "FLASK_ENV": cls.FLASK_ENV,
            "MONGODB_DB": cls.MONGODB_DB,
            "VECTOR_PROVIDER": cls.VECTOR_PROVIDER,
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL,
            "DEFAULT_LLM_MODEL": cls.DEFAULT_LLM_MODEL,
            "MAX_SOURCES_PER_CLAIM": cls.MAX_SOURCES_PER_CLAIM,
            "SIMILARITY_THRESHOLD": cls.SIMILARITY_THRESHOLD,
            "RATE_LIMIT_REQUESTS": cls.RATE_LIMIT_REQUESTS,
            "ENABLE_MULTIMODAL": cls.ENABLE_MULTIMODAL
        }
