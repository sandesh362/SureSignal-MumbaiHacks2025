# backend/services/mongodb_service.py
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId

from config import Config

logger = logging.getLogger(__name__)


class MongoDBService:
    """
    Small wrapper around pymongo for the VeriPulse collections used by Orchestrator.
    Collections used:
      - evidence
      - claims
      - verifications
      - bot_interactions
    """

    def __init__(self):
        uri = getattr(Config, "MONGODB_URI", None)
        if not uri:
            raise ValueError("MONGODB_URI must be set in Config or environment")
        self.client = MongoClient(uri)
        self.db = self.client.get_database(getattr(Config, "MONGODB_DB", "veripulse"))
        self._ensure_indexes()

    def _ensure_indexes(self):
        try:
            self.db.evidence.create_index([("url", ASCENDING)], unique=True, partialFilterExpression={"url": {"$exists": True}})
            self.db.claims.create_index([("claim_text", ASCENDING)])
            self.db.verifications.create_index([("claim_id", ASCENDING)])
            self.db.bot_interactions.create_index([("user_id", ASCENDING)])
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")

    def store_evidence(self, evidence: Dict) -> str:
        """Insert evidence doc and return string id"""
        evidence.setdefault("created_at", datetime.utcnow())
        try:
            res = self.db.evidence.insert_one(evidence)
            return str(res.inserted_id)
        except Exception:
            # duplicate (by url) -> try to find and return existing id
            q = {"url": evidence.get("url")}
            existing = self.db.evidence.find_one(q)
            if existing:
                return str(existing["_id"])
            raise

    def store_claim(self, claim: Dict) -> str:
        claim.setdefault("timestamp", datetime.utcnow())
        res = self.db.claims.insert_one(claim)
        return str(res.inserted_id)

    def store_verification(self, verification: Dict) -> str:
        verification.setdefault("timestamp", datetime.utcnow())
        res = self.db.verifications.insert_one(verification)
        return str(res.inserted_id)

    def store_bot_interaction(self, interaction: Dict) -> str:
        interaction.setdefault("timestamp", datetime.utcnow())
        res = self.db.bot_interactions.insert_one(interaction)
        return str(res.inserted_id)

    def get_user_rate_limit(self, user_id: str, window_minutes: int = 60) -> int:
        """Return count of interactions from user in the recent window."""
        window = datetime.utcnow() - timedelta(minutes=window_minutes)
        return self.db.bot_interactions.count_documents({"user_id": user_id, "timestamp": {"$gte": window}})

    def get_verification_history(self, claim_id: str) -> List[Dict]:
        """Return verifications for a claim id if present (most recent first)"""
        if not claim_id:
            return []
        try:
            oid = ObjectId(claim_id)
        except Exception:
            return []
        cursor = self.db.verifications.find({"claim_id": claim_id}).sort("timestamp", DESCENDING).limit(5)
        return list(cursor)

    def get_statistics(self) -> Dict:
        try:
            evidence_count = self.db.evidence.count_documents({})
            claims_count = self.db.claims.count_documents({})
            verifications_count = self.db.verifications.count_documents({})
            return {
                "evidence_count": evidence_count,
                "claims_count": claims_count,
                "verifications_count": verifications_count
            }
        except Exception as e:
            logger.warning(f"Error fetching stats: {e}")
            return {}

    # Expose db for direct queries (used in your app / health checks)
    @property
    def raw_db(self):
        return self.db
