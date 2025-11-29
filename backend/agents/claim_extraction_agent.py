# backend/agents/claim_extraction_agent.py
from typing import List, Dict
from datetime import datetime
import re

class ClaimExtractionAgent:
    """
    Minimal claim extraction & clustering stub to allow local testing.
    - process_batch(posts) -> returns {'trending': [ {id, claim_text, score, first_seen}, ... ]}
    """

    def __init__(self):
        pass

    def _extract_claim_from_text(self, text: str) -> str:
        # Very naive heuristic: pick the longest sentence or the first sentence with a numeric/location word
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return text[:250]
        # choose the most "assertive" looking sentence (longest)
        cand = max(sentences, key=lambda s: len(s))
        return cand.strip()

    def process_batch(self, posts: List[Dict]) -> Dict:
        """
        Accepts list of posts (as produced by IngestionAgent) and returns
        a dict with a 'trending' key that is a list of claim dicts.
        """
        clusters = {}
        for p in posts:
            text = p.get("text", "")
            claim_text = self._extract_claim_from_text(text)
            key = claim_text[:160]  # primitive dedupe key
            if key not in clusters:
                clusters[key] = {
                    "id": f"claim_{len(clusters)+1}",
                    "claim_text": claim_text,
                    "count": 0,
                    "first_seen": p.get("timestamp")
                }
            clusters[key]["count"] += 1

        # convert to list and sort by count (descending)
        trending = sorted(clusters.values(), key=lambda x: x["count"], reverse=True)
        # attach a simple 'score'
        for i, t in enumerate(trending):
            t["score"] = round(1.0 - (i * 0.1), 2)

        return {"trending": trending}
