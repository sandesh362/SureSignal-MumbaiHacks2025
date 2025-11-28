"""
TruthGuard - Advanced AI Misinformation Detection System
Complete Backend Implementation with Multi-Agent Architecture
Supports: OpenAI (ChatGPT), Google Gemini, Anthropic Claude
"""

import asyncio
import json
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from textblob import TextBlob
import numpy as np

from typing import List, Optional
from content_ingestion import ContentIngestionClient

# --- .env loading ---
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# AI PROVIDER DETECTION
# ============================================================================

AI_PROVIDER = None
AI_CLIENT = None

try:
    import openai

    if os.getenv("OPENAI_API_KEY"):
        AI_PROVIDER = "openai"
        print("✓ Using OpenAI (ChatGPT)")
except ImportError:
    pass

if not AI_PROVIDER:
    try:
        import google.generativeai as genai

        if os.getenv("GOOGLE_API_KEY"):
            AI_PROVIDER = "gemini"
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            print("✓ Using Google Gemini")
    except ImportError:
        pass

if not AI_PROVIDER:
    try:
        import anthropic

        if os.getenv("ANTHROPIC_API_KEY"):
            AI_PROVIDER = "anthropic"
            print("✓ Using Anthropic Claude")
    except ImportError:
        pass

# ============================================================================
# CONFIGURATION & MODELS
# ============================================================================


class ClaimStatus(str, Enum):
    DETECTED = "detected"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FALSE = "false"
    MISLEADING = "misleading"
    UNVERIFIABLE = "unverifiable"


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Source:
    """Represents an information source"""

    url: str
    title: str
    domain: str
    credibility_score: float
    timestamp: datetime
    content: str
    source_type: str  # news, social, official, academic


@dataclass
class Claim:
    """Represents a claim to be verified"""

    id: str
    text: str
    sources: List[Source] = field(default_factory=list)
    status: ClaimStatus = ClaimStatus.DETECTED
    confidence: float = 0.0
    severity: SeverityLevel = SeverityLevel.MEDIUM
    detected_at: datetime = field(default_factory=datetime.now)
    verified_at: Optional[datetime] = None
    verification_summary: str = ""
    supporting_evidence: List[Dict] = field(default_factory=list)
    contradicting_evidence: List[Dict] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    viral_score: float = 0.0
    sentiment_score: float = 0.0
    entities: List[str] = field(default_factory=list)
    related_claims: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of claim verification"""

    claim_id: str
    status: ClaimStatus
    confidence: float
    summary: str
    evidence: List[Dict]
    context: Dict
    reasoning: str


# ============================================================================
# AI PROVIDER WRAPPER
# ============================================================================


"""
Complete Gemini Free Tier Configuration for TruthGuard
This configuration uses the free Gemini API with proper model names and rate limiting
"""

# ============================================================================
# STEP 1: Update the AIProvider class in your main.py
# ============================================================================

class AIProvider:
    """Universal AI provider wrapper - Optimized for Gemini Free Tier"""

    def __init__(self, api_key: str = None):
        global AI_PROVIDER, AI_CLIENT

        if AI_PROVIDER is None:
            raise RuntimeError(
                "No AI provider available. "
                "Set one of OPENAI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY "
                "and install the corresponding library."
            )

        self.provider = AI_PROVIDER

        if self.provider == "openai":
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = openai
            self.model = "gpt-4-turbo-preview"

        elif self.provider == "gemini":
            # Configure Gemini Free Tier
            if api_key:
                genai.configure(api_key=api_key)
            else:
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            # FREE TIER MODELS (Choose one):
            # Option 1: Latest Flash model (RECOMMENDED - Fast & Free)
            self.model = "gemini-1.5-flash-latest"
            
            # Option 2: Specific Flash version (More stable)
            # self.model = "gemini-1.5-flash-002"
            
            # Option 3: Gemini Pro (Older but very stable)
            # self.model = "gemini-pro"
            
            # Option 4: Experimental free model
            # self.model = "gemini-1.5-flash-8b-latest"  # Even faster, lighter
            
            print(f"  Using Gemini Model: {self.model}")
            
            # Verify model availability
            try:
                available_models = [m.name for m in genai.list_models() 
                                  if 'generateContent' in m.supported_generation_methods]
                if f"models/{self.model}" not in available_models and self.model not in available_models:
                    print(f"  ⚠️  Warning: {self.model} not found in available models")
                    print(f"  Available models: {available_models[:3]}")
                    # Fallback to most stable free model
                    self.model = "gemini-1.5-flash-latest"
                    print(f"  Falling back to: {self.model}")
            except Exception as e:
                print(f"  Could not verify model availability: {e}")

        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            self.model = "claude-3-5-sonnet-20240620"

        else:
            raise RuntimeError(
                f"AI_PROVIDER set to unknown value: {self.provider}. "
                "Expected one of: openai, gemini, anthropic."
            )

    async def generate(
        self, prompt: str, system_prompt: str = "", use_search: bool = False
    ) -> str:
        """Universal generation method with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return await self._openai_generate(prompt, system_prompt, use_search)
                elif self.provider == "gemini":
                    return await self._gemini_generate(prompt, system_prompt, use_search)
                elif self.provider == "anthropic":
                    return await self._anthropic_generate(prompt, system_prompt, use_search)
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limiting for free tier
                if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"  Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"  Max retries reached. Rate limit error: {e}")
                        return ""
                
                # Handle 404 model not found
                elif "404" in error_msg or "not found" in error_msg:
                    print(f"  Model error: {e}")
                    if self.provider == "gemini" and attempt == 0:
                        # Try fallback model
                        old_model = self.model
                        self.model = "gemini-1.5-flash-latest"
                        print(f"  Trying fallback model: {self.model}")
                        continue
                    return ""
                
                else:
                    print(f"  AI Generation Error: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    return ""
        
        return ""

    async def _gemini_generate(
        self, prompt: str, system_prompt: str, use_search: bool
    ) -> str:
        """Gemini generation optimized for free tier"""
        try:
            # Combine prompts for Gemini (it doesn't have separate system prompt in free tier)
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            # Create model with safety settings optimized for fact-checking
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,  # Free tier limit
            }
            
            # Safety settings (adjust as needed)
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
            
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Generate content (synchronous call wrapped in async)
            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts') and response.parts:
                text_parts = []
                for part in response.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                return ''.join(text_parts)
            else:
                # Check if blocked by safety filters
                if hasattr(response, 'prompt_feedback'):
                    print(f"  Response blocked: {response.prompt_feedback}")
                return ""
                
        except Exception as e:
            error_str = str(e)
            if "quota" in error_str.lower():
                print(f"  ⚠️  Gemini quota exceeded. Free tier limits:")
                print(f"     • 15 requests per minute")
                print(f"     • 1,500 requests per day")
                print(f"     • 1 million tokens per day")
            raise  # Re-raise to be handled by retry logic


# ============================================================================
# STEP 2: Add rate limiting helper (Optional but recommended)
# ============================================================================

class RateLimiter:
    """Simple rate limiter for Gemini Free Tier"""
    
    def __init__(self, max_requests_per_minute: int = 15):
        self.max_requests = max_requests_per_minute
        self.requests = []
    
    async def wait_if_needed(self):
        """Wait if we're about to exceed rate limit"""
        now = datetime.now()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests 
                        if (now - req_time).total_seconds() < 60]
        
        if len(self.requests) >= self.max_requests:
            # Wait until oldest request is > 1 minute old
            oldest = self.requests[0]
            wait_time = 60 - (now - oldest).total_seconds()
            if wait_time > 0:
                print(f"  Rate limit: waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
        
        self.requests.append(now)


# ============================================================================
# STEP 3: Update AgentOrchestrator to use rate limiting
# ============================================================================

class AgentOrchestrator:
    """Coordinates all AI agents with rate limiting"""

    def __init__(self, api_key: str = None, default_topic: str | None = None):
        self.ai_provider = AIProvider(api_key)
        
        # Add rate limiter for free tier
        self.rate_limiter = RateLimiter(max_requests_per_minute=12)  # Conservative limit
        
        # AI agents
        self.scanner = ScanningAgent(self.ai_provider)
        self.detector = DetectionAgent(self.ai_provider)
        self.verifier = VerificationAgent(self.ai_provider)
        self.analyzer = AnalysisAgent(self.ai_provider)
        self.communicator = CommunicationAgent(self.ai_provider)

        # ... rest of your initialization ...
        self.claims: Dict[str, Claim] = {}
        self.claim_history: List[Dict] = []
        self.default_topic: str | None = default_topic
        self.default_audience: str = "general_public"
        
        self.stats = {
            "claims_scanned": 0,
            "claims_verified": 0,
            "misinformation_detected": 0,
            "sources_monitored": len(self.scanner.monitored_sources)
            if hasattr(self.scanner, "monitored_sources")
            else 0,
            "avg_verify_time": 0.0,
        }


# ============================================================================
# STEP 4: Update BaseAgent to use rate limiter
# ============================================================================

class BaseAgent:
    """Base class for all AI agents with rate limiting"""

    def __init__(self, name: str, ai_provider: AIProvider):
        self.name = name
        self.ai = ai_provider
        self.rate_limiter = RateLimiter(max_requests_per_minute=12)

    async def think(
        self, prompt: str, system_prompt: str = "", use_search: bool = False
    ) -> str:
        """Core reasoning method using AI provider with rate limiting"""
        try:
            # Wait if needed to respect rate limits
            await self.rate_limiter.wait_if_needed()
            
            return await self.ai.generate(prompt, system_prompt, use_search)
        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            return ""


print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           ✅ GEMINI FREE TIER CONFIGURATION READY             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Free Tier Limits:
• 15 requests per minute
• 1,500 requests per day  
• 1 million tokens per day
• 2048 max output tokens per request

Recommended Models:
✓ gemini-1.5-flash-latest (Best for production)
✓ gemini-1.5-flash-8b-latest (Faster, lighter)
✓ gemini-pro (Most stable)

Rate limiting enabled to prevent quota errors!
""")

# ============================================================================
# ADVANCED AI AGENTS
# ============================================================================


class BaseAgent:
    """Base class for all AI agents"""

    def __init__(self, name: str, ai_provider: AIProvider):
        self.name = name
        self.ai = ai_provider

    async def think(
        self, prompt: str, system_prompt: str = "", use_search: bool = False
    ) -> str:
        """Core reasoning method using AI provider"""
        try:
            return await self.ai.generate(prompt, system_prompt, use_search)
        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            return ""


class ScanningAgent(BaseAgent):
    """
    Monitors multiple sources and detects emerging claims.

    Uses ContentIngestionClient to fetch REAL, topic-focused text
    from external APIs (news, social, forums, etc).
    """

    def __init__(self, ai_provider: AIProvider):
        super().__init__("ScanningAgent", ai_provider)
        self.monitored_sources = [
            "social_media",
            "news_outlets",
            "forums",
            "messaging_platforms",
            "government_sites",
        ]
        self.ingestion = ContentIngestionClient()

    async def scan_sources(
        self,
        source_type: str,
        topic: Optional[str] = None,
    ) -> List[str]:
        """
        Fetch real text from external sources and ask the LLM to extract
        concrete factual claims that may be misinformation.

        :param source_type: one of self.monitored_sources
        :param topic: optional topic to narrow down the scan
        :return: list of canonical claim strings
        """
        system_prompt = """You are a scanning agent that identifies potential misinformation claims.
        Analyze content and extract specific claims that could be misinformation.
        Return only factual claims that can be verified, not opinions or vague predictions."""

        # 1) Fetch text snippets from ingestion layer / external APIs
        snippets = await self.ingestion.fetch_corpus(
            source_type=source_type,
            topic=topic,
            limit=30,
        )

        if not snippets:
            return []

        # Build a corpus text
        content = "\n\n---\n\n".join(snippets)
        content = content[:16000]  # safety truncation

        topic_part = (
            f"\nFocus ONLY on claims related to this topic: '{topic}'."
            if topic
            else ""
        )

        prompt = f"""You will be given a batch of real-world content from {source_type}.
Each post/article/thread may contain one or more factual claims.

Your job:
- Extract specific factual claims that could be checked or fact-checked.
- Ignore jokes, obvious satire, pure opinions, or vague speculation.
- Merge duplicates into a single canonical phrasing.

Content:
{content}

{topic_part}

Return 3–15 claims as a JSON array, for example:
[
  {{"claim": "specific claim text", "priority": "high/medium/low"}}
]
"""

        response = await self.think(prompt, system_prompt)

        try:
            claims_data = json.loads(response.strip())
        except Exception:
            return []

        claims: List[str] = []
        for c in claims_data:
            if isinstance(c, dict) and "claim" in c:
                text = (c["claim"] or "").strip()
                if text:
                    claims.append(text)
        return claims

    async def detect_viral_patterns(self, claim_text: str) -> float:
        """
        Given a single claim, estimate how likely it is to go viral,
        based on its content alone.
        """
        system_prompt = """You are an expert at predicting viral content spread.
Analyze claims for characteristics that make them go viral."""

        prompt = f"""Analyze this claim for viral potential (0–100 score):

Claim: {claim_text}

Consider:
- Emotional intensity
- Shock / fear / outrage factor
- Political or identity triggers
- Simplicity and shareability
- Past patterns of similar claims

Return ONLY a number from 0 to 100, no explanation."""

        response = await self.think(prompt, system_prompt)
        try:
            return float(response.strip()) / 100
        except Exception:
            return 0.5
        

class DetectionAgent(BaseAgent):
    """Analyzes claims and detects misinformation patterns"""

    def __init__(self, ai_provider: AIProvider):
        super().__init__("DetectionAgent", ai_provider)

    # --------------- internal helpers -----------------

    def _safe_parse_json_obj(self, text: str) -> Dict:
        """
        Try to be tolerant with the LLM output:
        - Strip whitespace
        - Try direct json.loads
        - If that fails, try to extract the first {...} block with regex
        - If that fails, return {}
        """
        text = (text or "").strip()
        if not text:
            return {}

        # 1) direct attempt
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) try to extract a JSON object
        try:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return {}

    def _safe_parse_json_array(self, text: str):
        """
        Same as above but expects an array.
        """
        text = (text or "").strip()
        if not text:
            return []

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            # some models may wrap it as {"entities": [...]}
            if isinstance(data, dict):
                if "entities" in data and isinstance(data["entities"], list):
                    return data["entities"]
        except Exception:
            pass

        try:
            match = re.search(r"\[[\s\S]*\]", text)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return []

    # --------------- main APIs -----------------

    async def analyze_claim(
        self,
        claim: Claim,
        topic: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Dict:
        """
        Deep analysis of a claim to understand:
        - Suspicion level
        - Red flags / manipulation patterns
        - Claim type (factual / prediction / opinion / mixed)
        - Verification complexity
        - Recommended sources and actions

        `topic` and `source_type` are optional but make the analysis more contextual.
        """
        system_prompt = """You are an expert misinformation detector with deep knowledge of:
        - Common misinformation patterns and tactics
        - Logical fallacies and manipulation techniques
        - Verification methodologies
        - Source credibility assessment

        Your job is to evaluate individual claims and describe:
        - How suspicious they are as potential misinformation
        - Why (concrete patterns, not vague impressions)
        - How hard it will be to verify them
        - What sources and next steps are appropriate.

        Be conservative: "high" suspicion should be reserved for claims that
        clearly match known misinformation patterns or have multiple red flags.
        """

        topic_part = f"\nContext topic: {topic}" if topic else ""
        source_part = f"\nSource type: {source_type}" if source_type else ""

        prompt = f"""Analyze this claim for misinformation indicators:

Claim: {claim.text}
Detected at: {claim.detected_at.isoformat()}
Current severity (if any): {getattr(claim, "severity", None)}
Viral score (0–1): {getattr(claim, "viral_score", 0.0)}
Sentiment score (TextBlob polarity, -1 to 1): {getattr(claim, "sentiment_score", 0.0)}
Existing entities (if already extracted): {', '.join(claim.entities) if claim.entities else 'None'}
{topic_part}{source_part}

Provide analysis in JSON format with the following fields:

{{
  "suspicion_level": "high" | "medium" | "low",
  "primary_concern": "short summary of the main reason for suspicion",
  "red_flags": [
    "list of concrete concerning patterns (e.g., 'extraordinary claim without evidence', 'appeal to fear', ...)"
  ],
  "manipulation_techniques": [
    "list of rhetorical / emotional manipulation techniques if any (e.g., 'fearmongering', 'scapegoating', 'whataboutism')"
  ],
  "emotional_tone": "neutral" | "fear" | "anger" | "outrage" | "hope" | "mixed",
  "claim_type": "factual" | "prediction" | "opinion" | "mixed",
  "verification_complexity": "simple" | "moderate" | "complex",
  "recommended_sources": [
    "types of sources to check (e.g., 'official government releases', 'peer-reviewed medical studies', 'independent fact-checkers')"
  ],
  "recommended_actions": [
    "suggested next actions (e.g., 'flag_for_human_review', 'low_priority_monitoring', 'urgent_fact_check')"
  ]
}}

Rules:
- Stick to these keys and enumerated values where specified.
- Do NOT include any explanatory text outside the JSON.
"""

        response = await self.think(prompt, system_prompt)

        data = self._safe_parse_json_obj(response)

        # Provide safe defaults if the model returned something incomplete
        if not data:
            data = {}

        suspicion = data.get("suspicion_level") or "medium"
        if suspicion not in ("high", "medium", "low"):
            suspicion = "medium"

        claim_type = data.get("claim_type") or "factual"
        if claim_type not in ("factual", "prediction", "opinion", "mixed"):
            claim_type = "factual"

        verification_complexity = data.get("verification_complexity") or "moderate"
        if verification_complexity not in ("simple", "moderate", "complex"):
            verification_complexity = "moderate"

        # Normalize lists
        red_flags = data.get("red_flags") or []
        if not isinstance(red_flags, list):
            red_flags = [str(red_flags)]

        manipulation_techniques = data.get("manipulation_techniques") or []
        if not isinstance(manipulation_techniques, list):
            manipulation_techniques = [str(manipulation_techniques)]

        recommended_sources = data.get("recommended_sources") or []
        if not isinstance(recommended_sources, list):
            recommended_sources = [str(recommended_sources)]

        recommended_actions = data.get("recommended_actions") or []
        if not isinstance(recommended_actions, list):
            recommended_actions = [str(recommended_actions)]

        emotional_tone = data.get("emotional_tone") or "neutral"

        cleaned = {
            "suspicion_level": suspicion,
            "primary_concern": data.get("primary_concern", ""),
            "red_flags": red_flags,
            "manipulation_techniques": manipulation_techniques,
            "emotional_tone": emotional_tone,
            "claim_type": claim_type,
            "verification_complexity": verification_complexity,
            "recommended_sources": recommended_sources,
            "recommended_actions": recommended_actions,
        }

        return cleaned

    async def extract_entities(self, text: str) -> List[str]:
        """
        Extract key entities from the text.
        We keep the public API returning List[str] for backward compatibility,
        but internally we let the model return richer structure if it wants.

        Example desired output from the model:

        [
          {"name": "Reserve Bank of India", "type": "organization"},
          {"name": "Mumbai", "type": "location"}
        ]

        or just:
        ["Reserve Bank of India", "Mumbai"]
        """
        prompt = f"""Extract key entities (people, organizations, locations, events, institutions)
from the following text:

{text}

Return your answer in JSON.

You may use either of these formats:

1) Simple list of names:
[
  "Entity 1",
  "Entity 2",
  ...
]

2) Enriched list of objects:
[
  {{"name": "Entity 1", "type": "organization"}},
  {{"name": "Entity 2", "type": "location"}}
]

Rules:
- Do NOT include any explanation or text outside the JSON.
- Ignore generic words like 'government', 'people', 'they', 'experts' unless
  they refer to a specific, identifiable entity.
"""

        response = await self.think(prompt)

        raw = self._safe_parse_json_array(response)

        entities: List[str] = []

        # If it's a list of strings
        if all(isinstance(x, str) for x in raw):
            entities = [x.strip() for x in raw if x.strip()]
        # If it's a list of dicts with "name"
        elif all(isinstance(x, dict) for x in raw):
            for item in raw:
                name = (item.get("name") or item.get("entity") or "").strip()
                if name:
                    entities.append(name)

        # Deduplicate while preserving order
        seen = set()
        unique_entities: List[str] = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique_entities.append(e)

        return unique_entities


class VerificationAgent(BaseAgent):
    """Verifies claims against trusted sources and structures the evidence."""

    def __init__(self, ai_provider: AIProvider):
        super().__init__("VerificationAgent", ai_provider)

    # --------------- internal helpers -----------------

    def _safe_parse_json_obj(self, text: str) -> Dict:
        """
        Try to parse a JSON object from model output, even if it has
        extra text before/after. Fallback to {} on failure.
        """
        text = (text or "").strip()
        if not text:
            return {}

        # 1) direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) extract first {...} block
        try:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return {}

    def _normalize_status(self, raw_status: Optional[str]) -> ClaimStatus:
        """
        Map arbitrary status strings from the model into your ClaimStatus enum.
        Accepts common synonyms and normalizes them.
        """
        if not raw_status:
            return ClaimStatus.UNVERIFIABLE

        s = str(raw_status).strip().lower()

        # direct matches
        if s in ("verified", "true", "correct", "accurate"):
            return ClaimStatus.VERIFIED
        if s in ("false", "fake", "incorrect", "debunked"):
            return ClaimStatus.FALSE
        if s in ("misleading", "partly_true", "partly false", "half true", "mixed"):
            return ClaimStatus.MISLEADING
        if s in ("unverifiable", "unknown", "insufficient_evidence", "uncertain"):
            return ClaimStatus.UNVERIFIABLE
        if s in ("detected", "flagged"):
            return ClaimStatus.DETECTED
        if s in ("verifying", "in_progress", "pending"):
            return ClaimStatus.VERIFYING

        # fallback
        return ClaimStatus.UNVERIFIABLE

    def _normalize_confidence(self, raw_conf) -> float:
        """
        Normalize model confidence to a float in [0, 1].
        Accepts either 0–1 or 0–100 style values.
        """
        try:
            val = float(raw_conf)
        except Exception:
            return 0.5

        # if they gave 0–100, scale down
        if val > 1.0:
            val = val / 100.0

        # clamp
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        return val

    def _normalize_evidence_list(self, lst) -> List[Dict]:
        """
        Normalize supporting / contradicting evidence arrays into a list of dicts
        with stable keys: source, evidence, credibility, direction.
        """
        if not isinstance(lst, list):
            return []

        normalized = []
        for item in lst:
            if isinstance(item, dict):
                source = (item.get("source") or item.get("name") or "Unknown").strip()
                evidence_text = (item.get("evidence") or item.get("statement") or "").strip()
                credibility = item.get("credibility", item.get("score", 0.5))
            else:
                # e.g., plain string
                source = "Unknown"
                evidence_text = str(item).strip()
                credibility = 0.5

            try:
                credibility = float(credibility)
            except Exception:
                credibility = 0.5

            if not evidence_text:
                continue

            normalized.append(
                {
                    "source": source,
                    "evidence": evidence_text,
                    "credibility": max(0.0, min(1.0, credibility)),
                }
            )
        return normalized

    # --------------- main API -----------------

    async def verify_claim(
        self,
        claim: Claim,
        detection: Optional[Dict] = None,
        topic: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a claim using the underlying LLM (optionally with web search if
        Anthropic is used).

        - `detection`: optional output from DetectionAgent.analyze_claim
          (suspicion_level, red_flags, etc.)
        - `topic`: optional topic string for extra context
        - `source_type`: where the claim came from (social_media, news_outlets, ...)

        For backward compatibility, orchestrator can still call:
            await self.verify_claim(claim)
        """
        system_prompt = """You are a professional fact-checker with expertise in:
        - Cross-referencing multiple independent sources
        - Identifying credible vs. low-credibility evidence
        - Detecting manipulated, cherry-picked, or out-of-context information
        - Providing balanced, evidence-based assessments

        Your goal is to:
        - Check whether a claim is supported, contradicted, or currently unverifiable
        - Summarize the evidence in a neutral, precise way
        - Clearly explain limitations or uncertainty when applicable.
        """

        detection_summary = ""
        if detection:
            detection_summary = f"""
Prior automated analysis:
- Suspicion level: {detection.get('suspicion_level')}
- Red flags: {', '.join(detection.get('red_flags', []))}
- Manipulation techniques: {', '.join(detection.get('manipulation_techniques', []))}
- Recommended sources: {', '.join(detection.get('recommended_sources', []))}
"""

        entities_str = ", ".join(claim.entities) if claim.entities else "None identified"
        topic_part = f"\nInvestigation topic: {topic}" if topic else ""
        source_part = f"\nClaim source type: {source_type}" if source_type else ""

        prompt = f"""You must verify the following claim and produce a structured fact-check.

Claim text:
\"\"\"{claim.text}\"\"\"

Key entities involved: {entities_str}
Detected at: {claim.detected_at.isoformat()}
Viral score (0–1): {getattr(claim, "viral_score", 0.0)}
Current severity label (if any): {getattr(claim, "severity", None)}
Text sentiment score (-1 to 1): {getattr(claim, "sentiment_score", 0.0)}
{topic_part}{source_part}
{detection_summary}

Tasks:
1. Determine the best current assessment of the claim's veracity.
2. Identify the strongest supporting and contradicting evidence from credible sources.
3. Assign a numeric confidence score (0–100).
4. Explain your reasoning, including limitations or remaining uncertainty.
5. Provide helpful context (historical background, related facts, nuance).

Use the following JSON format for your answer. Do NOT include any extra text outside this JSON:

{{
  "status": "verified" | "false" | "misleading" | "unverifiable",
  "confidence": 0–100,
  "summary": "2–3 sentence summary of the fact-check conclusion in neutral language.",
  "supporting_evidence": [
    {{
      "source": "source name or URL",
      "evidence": "short quote or paraphrase that supports the claim",
      "credibility": 0.0–1.0
    }}
  ],
  "contradicting_evidence": [
    {{
      "source": "source name or URL",
      "evidence": "short quote or paraphrase that contradicts or undermines the claim",
      "credibility": 0.0–1.0
    }}
  ],
  "reasoning": "Detailed explanation of how you arrived at the conclusion, including how you weighed conflicting evidence.",
  "context": {{
    "historical_background": "brief background that helps understand the claim",
    "related_facts": [
      "related fact 1",
      "related fact 2"
    ],
    "caveats": [
      "important limitation or uncertainty, if any"
    ]
  }}
}}"""

        try:
            # Only Anthropic has native web search tools in this setup
            use_search = self.ai.provider == "anthropic"

            response = await self.think(prompt, system_prompt, use_search=use_search)

            # ---------- parse JSON ----------
            result_data = self._safe_parse_json_obj(response)

            if not result_data:
                # fallback if model output was unusable
                result_data = {
                    "status": "unverifiable",
                    "confidence": 50,
                    "summary": "Unable to verify claim with the available information.",
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "reasoning": "Parsing or generation issue while attempting verification.",
                    "context": {
                        "historical_background": "",
                        "related_facts": [],
                        "caveats": ["Automated system could not parse a structured verification response."],
                    },
                }

            # ---------- normalize fields ----------
            status = self._normalize_status(result_data.get("status"))
            confidence = self._normalize_confidence(result_data.get("confidence", 50))

            summary = result_data.get("summary", "") or ""
            reasoning = result_data.get("reasoning", "") or ""

            supporting_raw = result_data.get("supporting_evidence", []) or []
            contradicting_raw = result_data.get("contradicting_evidence", []) or []

            supporting = self._normalize_evidence_list(supporting_raw)
            contradicting = self._normalize_evidence_list(contradicting_raw)

            context = result_data.get("context") or {}
            if not isinstance(context, dict):
                context = {}

            # embed structured evidence into context for downstream UI/analysis
            context.setdefault("structured_evidence", {})
            context["structured_evidence"]["supporting"] = supporting
            context["structured_evidence"]["contradicting"] = contradicting

            # combined evidence list for backward compatibility
            combined_evidence = []
            for ev in supporting:
                ev_copy = dict(ev)
                ev_copy["direction"] = "supporting"
                combined_evidence.append(ev_copy)
            for ev in contradicting:
                ev_copy = dict(ev)
                ev_copy["direction"] = "contradicting"
                combined_evidence.append(ev_copy)

            # ---------- build result object ----------
            return VerificationResult(
                claim_id=claim.id,
                status=status,
                confidence=confidence,
                summary=summary,
                evidence=combined_evidence,
                context=context,
                reasoning=reasoning,
            )

        except Exception as e:
            # Hard failure fallback
            print(f"[VerificationAgent] Error: {e}")
            return VerificationResult(
                claim_id=claim.id,
                status=ClaimStatus.UNVERIFIABLE,
                confidence=0.0,
                summary="Verification failed due to an internal error.",
                evidence=[],
                context={
                    "error": str(e),
                    "historical_background": "",
                    "related_facts": [],
                    "caveats": ["Internal exception in verification agent."],
                },
                reasoning=str(e),
            )


class AnalysisAgent(BaseAgent):
    """Performs deep impact analysis on (already) verified claims."""

    def __init__(self, ai_provider: AIProvider):
        super().__init__("AnalysisAgent", ai_provider)

    # --------------- helpers -----------------

    def _safe_parse_json_obj(self, text: str) -> Dict:
        """
        Try to parse a JSON object from the model's output, even if there is
        extra text around it.
        """
        text = (text or "").strip()
        if not text:
            return {}

        # direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # extract first {...}
        try:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group())
        except Exception:
            pass

        return {}

    def _normalize_severity(self, raw: Optional[str]) -> str:
        """
        Normalize severity to one of: critical, high, medium, low.
        """
        if not raw:
            return "medium"
        s = str(raw).strip().lower()

        if s in ("critical", "severe", "very_high", "urgent"):
            return "critical"
        if s in ("high", "elevated"):
            return "high"
        if s in ("medium", "moderate"):
            return "medium"
        if s in ("low", "minor"):
            return "low"

        return "medium"

    def _normalize_score01(self, raw, default: float = 0.5) -> float:
        """
        Normalize any numeric input to [0, 1].
        Accepts 0–1 or 0–100.
        """
        try:
            val = float(raw)
        except Exception:
            return default

        # if 0–100 style, scale
        if val > 1.0:
            val = val / 100.0

        # clamp
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        return val

    def _infer_default_severity(self, claim: Claim, verification: VerificationResult) -> str:
        """
        Heuristic fallback if the model gives nonsense or nothing.
        Uses:
        - verification.status
        - verification.confidence
        - claim.viral_score
        - claim.sentiment_score
        """
        status = verification.status
        conf = verification.confidence or 0.5
        viral = claim.viral_score or 0.0
        sentiment = claim.sentiment_score or 0.0

        # Very strong, very viral false/misleading claims → critical
        if status in (ClaimStatus.FALSE, ClaimStatus.MISLEADING) and conf >= 0.8 and viral >= 0.7:
            return "critical"

        # False/misleading, reasonably viral → high
        if status in (ClaimStatus.FALSE, ClaimStatus.MISLEADING) and (viral >= 0.4 or conf >= 0.7):
            return "high"

        # Unverifiable but viral + strong emotion → high/medium
        if status == ClaimStatus.UNVERIFIABLE and viral >= 0.6 and abs(sentiment) >= 0.4:
            return "high"

        # Verified true, low sentiment, low viral → low
        if status == ClaimStatus.VERIFIED and viral < 0.3 and abs(sentiment) < 0.3:
            return "low"

        # Default
        return "medium"

    # --------------- main API -----------------

    async def assess_impact(
        self,
        claim: Claim,
        verification: VerificationResult,
        detection: Optional[Dict] = None,
        topic: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Dict:
        """
        Assess the potential real-world impact of this claim, given its
        verification status and propagation signals.

        Returns a dict including at least:
        - severity: "critical" | "high" | "medium" | "low"
        - harm_potential: 0–1
        - urgency: 0–1

        Extra fields:
        - spread_risk
        - affected_demographics
        - impact_domains
        - recommended_action
        - notes
        """
        system_prompt = """You are an expert in assessing the impact of misinformation (or highly charged claims) on society.

You should consider factors like:
- Who could be harmed (health, safety, democratic process, minority groups, financial stability, etc.)
- How far and how fast this type of claim can spread
- How emotionally triggering and polarizing the claim is
- Whether the claim has already been assessed as false, misleading, or unverifiable
- How urgent it is to intervene (debunk, clarify, throttle, or monitor)

Your output will be used to prioritize moderation, fact-checking, and alerts.
Be careful not to overreact to minor or low-reach claims.
"""

        detection_part = ""
        if detection:
            detection_part = f"""
Prior pattern analysis:
- Suspicion level: {detection.get('suspicion_level')}
- Red flags: {', '.join(detection.get('red_flags', []))}
- Manipulation techniques: {', '.join(detection.get('manipulation_techniques', []))}
- Emotional tone: {detection.get('emotional_tone')}
"""

        topic_part = f"\nInvestigation topic: {topic}" if topic else ""
        source_part = f"\nPrimary content source: {source_type}" if source_type else ""

        status_name = verification.status.value if isinstance(verification.status, ClaimStatus) else str(verification.status)
        confidence_pct = int(round((verification.confidence or 0.0) * 100))

        prompt = f"""Assess the real-world impact of the following claim:

Claim text:
\"\"\"{claim.text}\"\"\"

Verification status: {status_name}
Verification confidence: {confidence_pct}%
Summary of verification: {verification.summary}

Claim metrics:
- Viral score (0–1): {claim.viral_score}
- Sentiment score (-1 to 1): {claim.sentiment_score}
- Detected at: {claim.detected_at.isoformat()}
- Entities: {', '.join(claim.entities) if claim.entities else 'None'}
{topic_part}{source_part}
{detection_part}

Based on this, evaluate:

1. How severe is the potential harm if people believe and act on this claim?
2. How likely is it to spread further, given its content and current viral score?
3. Who is most likely to be affected (demographics / groups)?
4. How urgent is it to intervene (debunk, clarify, rate-limit, or monitor)?

Return a SINGLE JSON object, no extra text, with this schema:

{{
  "severity": "critical" | "high" | "medium" | "low",
  "harm_potential": 0.0-1.0,
  "spread_risk": 0.0-1.0,
  "urgency": 0.0-1.0,
  "affected_demographics": [
    "group or demographic 1",
    "group or demographic 2"
  ],
  "impact_domains": [
    "health" | "democracy" | "economy" | "public_trust" | "social_cohesion" | "other"
  ],
  "recommended_action": "immediate_alert" | "prioritized_fact_check" | "monitor" | "low_priority",
  "notes": "1-3 sentences explaining why you chose this severity and action."
}}
"""

        try:
            response = await self.think(prompt, system_prompt)

            data = self._safe_parse_json_obj(response)

            if not data:
                # Fallback: infer something reasonable
                sev = self._infer_default_severity(claim, verification)
                return {
                    "severity": sev,
                    "harm_potential": 0.5,
                    "spread_risk": claim.viral_score or 0.5,
                    "urgency": 0.5,
                    "affected_demographics": [],
                    "impact_domains": [],
                    "recommended_action": "monitor",
                    "notes": "Default impact assessment applied due to parsing failure.",
                }

            severity = self._normalize_severity(data.get("severity"))

            # If model gave something weird like severity=low but it's a strongly viral confirmed-false claim,
            # we can bump it up using our heuristic.
            heuristic_severity = self._infer_default_severity(claim, verification)

            # Choose the "max" between the two severities (critical > high > medium > low)
            order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if order.get(heuristic_severity, 1) > order.get(severity, 1):
                severity = heuristic_severity

            harm_potential = self._normalize_score01(data.get("harm_potential"), default=0.5)
            urgency = self._normalize_score01(data.get("urgency"), default=0.5)
            spread_risk = self._normalize_score01(data.get("spread_risk", claim.viral_score or 0.5), default=claim.viral_score or 0.5)

            affected = data.get("affected_demographics") or []
            if not isinstance(affected, list):
                affected = [str(affected)]

            impact_domains = data.get("impact_domains") or []
            if not isinstance(impact_domains, list):
                impact_domains = [str(impact_domains)]

            recommended_action = data.get("recommended_action") or "monitor"
            notes = data.get("notes") or ""

            result = {
                "severity": severity,
                "harm_potential": harm_potential,
                "spread_risk": spread_risk,
                "urgency": urgency,
                "affected_demographics": affected,
                "impact_domains": impact_domains,
                "recommended_action": recommended_action,
                "notes": notes,
            }

            return result

        except Exception as e:
            print(f"[AnalysisAgent] Error: {e}")
            # Last-resort fallback
            sev = self._infer_default_severity(claim, verification)
            return {
                "severity": sev,
                "harm_potential": 0.5,
                "spread_risk": claim.viral_score or 0.5,
                "urgency": 0.5,
                "affected_demographics": [],
                "impact_domains": [],
                "recommended_action": "monitor",
                "notes": f"Impact assessment failed due to internal error: {e}",
            }


class CommunicationAgent(BaseAgent):
    """Generates public-friendly communications for verified/assessed claims."""

    def __init__(self, ai_provider: AIProvider):
        super().__init__("CommunicationAgent", ai_provider)

    def _status_label_for_public(self, status: ClaimStatus) -> str:
        """
        Map internal ClaimStatus to a short, public-facing phrase.
        """
        if status == ClaimStatus.VERIFIED:
            return "Mostly accurate"
        if status == ClaimStatus.FALSE:
            return "False claim"
        if status == ClaimStatus.MISLEADING:
            return "Misleading / partly false"
        if status == ClaimStatus.UNVERIFIABLE:
            return "Currently unverified"
        if status == ClaimStatus.VERIFYING:
            return "Under review"
        return "Needs careful verification"

    def _severity_badge(self, impact: Optional[Dict]) -> str:
        """
        Turn impact['severity'] into a simple badge string.
        """
        if not impact:
            return "Medium impact"
        sev = (impact.get("severity") or "medium").lower()
        if sev == "critical":
            return "🔴 Critical impact"
        if sev == "high":
            return "🟠 High impact"
        if sev == "medium":
            return "🟡 Medium impact"
        if sev == "low":
            return "🟢 Low impact"
        return "Medium impact"

    def _recommended_action_label(self, impact: Optional[Dict]) -> str:
        if not impact:
            return "Monitor and share accurate information."
        action = (impact.get("recommended_action") or "").lower()
        mapping = {
            "immediate_alert": "Immediate alert and wide clarification.",
            "prioritized_fact_check": "Prioritized fact-check and clear explanation.",
            "monitor": "Monitor and address if it spreads further.",
            "low_priority": "Low-priority monitoring.",
        }
        return mapping.get(action, "Monitor and share accurate information.")

    def _build_evidence_summary(self, verification: VerificationResult) -> str:
        """
        Prefer structured evidence inside verification.context['structured_evidence']
        if available; otherwise, fall back to the first few items in verification.evidence.
        """
        ctx = verification.context or {}
        structured = ctx.get("structured_evidence") if isinstance(ctx, dict) else None

        items = []
        if structured and isinstance(structured, dict):
            sup = structured.get("supporting") or []
            con = structured.get("contradicting") or []
            # pick top 2 supporting, top 2 contradicting
            for e in sup[:2]:
                src = e.get("source", "Unknown")
                txt = e.get("evidence", "")
                if txt:
                    items.append(f"- ✅ {src}: {txt}")
            for e in con[:2]:
                src = e.get("source", "Unknown")
                txt = e.get("evidence", "")
                if txt:
                    items.append(f"- ❌ {src}: {txt}")
        else:
            for e in verification.evidence[:4]:
                src = e.get("source", "Unknown")
                txt = e.get("evidence", "")
                direction = e.get("direction", "").lower()
                icon = "✅" if direction == "supporting" else ("❌" if direction == "contradicting" else "•")
                if txt:
                    items.append(f"- {icon} {src}: {txt}")

        return "\n".join(items) if items else "- Evidence summary not available."

    async def generate_public_alert(
        self,
        claim: Claim,
        verification: VerificationResult,
        impact: Optional[Dict] = None,
        audience: str = "general_public",
    ) -> str:
        """
        Generate a markdown public alert that is:

        - Clear and accessible
        - Non-alarmist, non-technical
        - Tailored to the verification status and impact

        Returns a markdown string.
        """
        system_prompt = """You are an expert public communicator specializing in clear,
accessible explanations about misinformation and sensitive claims.

Your goals:
- Inform people without causing panic or shame.
- Use simple, respectful language.
- Emphasize evidence and uncertainty honestly.
- Encourage people to check trusted sources and avoid sharing unverified claims.

Avoid:
- Political advocacy
- Insults or judgmental tone
- Overconfidence when evidence is limited
"""

        status = verification.status
        status_label = self._status_label_for_public(status)
        severity_badge = self._severity_badge(impact)
        recommended_action = self._recommended_action_label(impact)
        confidence_pct = int(round((verification.confidence or 0.0) * 100))

        evidence_summary = self._build_evidence_summary(verification)
        historical_context = ""
        related_facts = []
        caveats = []

        ctx = verification.context or {}
        if isinstance(ctx, dict):
            hist = ctx.get("historical_background")
            if isinstance(hist, str):
                historical_context = hist
            rf = ctx.get("related_facts")
            if isinstance(rf, list):
                related_facts = [str(x) for x in rf if str(x).strip()]
            cv = ctx.get("caveats")
            if isinstance(cv, list):
                caveats = [str(x) for x in cv if str(x).strip()]

        severity = (impact.get("severity") if impact else None) or "medium"
        harm_potential = impact.get("harm_potential") if impact else None
        urgency = impact.get("urgency") if impact else None

        # Compact numeric summary for model to see
        impact_summary = f"Severity: {severity}; Harm potential: {harm_potential}; Urgency: {urgency}" if impact else "No detailed impact assessment available."

        # Audience hint (you could later vary tone: teens, seniors, journalists, etc.)
        audience_hint = {
            "general_public": "Write for a general audience with mixed levels of education.",
            "students": "Write for students in simple, friendly language.",
            "journalists": "Write for journalists and media professionals.",
        }.get(audience, "Write for a general audience with mixed levels of education.")

        prompt = f"""Create a concise, public-facing alert about this claim.

Claim:
\"\"\"{claim.text}\"\"\"

Internal assessment:
- Verification status (internal): {status.value if isinstance(status, ClaimStatus) else status}
- Public-facing status label: {status_label}
- Model confidence: {confidence_pct}%
- Impact summary: {impact_summary}
- Badge: {severity_badge}

Verification summary (for your reference):
{verification.summary}

Evidence (for your reference):
{evidence_summary}

Historical / contextual notes (if any):
{historical_context or "None"}

Related facts (if any):
{", ".join(related_facts) if related_facts else "None"}

Caveats and uncertainties (if any):
{", ".join(caveats) if caveats else "None"}

Recommended internal action:
{recommended_action}

{audience_hint}

Now write a short alert in MARKDOWN with the following sections:

# 1. Headline
- One line, informative, not sensational.

## 2. What we found
- 2–4 short sentences explaining whether the claim is accurate, false, misleading, or currently unverified.
- Make clear how certain this assessment is (e.g., "Based on current evidence..." or "We do not have enough reliable information yet...").

## 3. Key facts
- Bullet list (3–6 bullets) summarizing the most important facts or clarifications in neutral language.

## 4. What you should do
- 2–4 bullets telling people how to behave around this claim:
  - whether to share or not
  - where to check for official updates
  - how to think about similar claims in the future

## 5. Sources and notes
- Brief mention of the types of sources used (e.g., official statements, reputable news outlets, independent fact-checkers).
- One short line about uncertainty or limitations if applicable.

Word limit:
- Aim for 120–220 words total.
- Be calm, respectful, and practical.
"""

        response = await self.think(prompt, system_prompt)
        return response


# ============================================================================
# ORCHESTRATION SYSTEM
# ============================================================================


class AgentOrchestrator:
    """Coordinates all AI agents"""

    def __init__(self, api_key: str = None, default_topic: str | None = None):
        self.ai_provider = AIProvider(api_key)

        # AI agents
        self.scanner = ScanningAgent(self.ai_provider)
        self.detector = DetectionAgent(self.ai_provider)
        self.verifier = VerificationAgent(self.ai_provider)
        self.analyzer = AnalysisAgent(self.ai_provider)
        self.communicator = CommunicationAgent(self.ai_provider)

        # Storage
        self.claims: Dict[str, Claim] = {}
        self.claim_history: List[Dict] = []

        # Config
        self.default_topic: str | None = default_topic
        self.default_audience: str = "general_public"

        # Stats
        self.stats = {
            "claims_scanned": 0,
            "claims_verified": 0,
            "misinformation_detected": 0,
            "sources_monitored": len(self.scanner.monitored_sources)
            if hasattr(self.scanner, "monitored_sources")
            else 0,
            "avg_verify_time": 0.0,  # seconds, rolling average
        }

    def generate_claim_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    async def process_claim(
        self,
        claim_text: str,
        *,
        topic: Optional[str] = None,
        source_type: Optional[str] = None,
        audience: str | None = None,
    ) -> Claim:
        """
        Full processing pipeline for a single claim string.

        - topic: optional context topic (e.g., "indian elections 2024")
        - source_type: where this claim came from (social_media, news_outlets, ...)
        - audience: target audience for the public alert (default: general_public)
        """
        claim_id = self.generate_claim_id(claim_text)

        # If already processed, you *could* return cached claim here
        # but for now we always re-process.
        claim = Claim(
            id=claim_id,
            text=claim_text,
            detected_at=datetime.now(),
        )

        if topic is None:
            topic = self.default_topic
        if audience is None:
            audience = self.default_audience

        print(f"\n{'=' * 60}")
        print(f"Processing Claim: {claim_text[:60]}...")
        print(f"{'=' * 60}")

        # ------------------------------------------------------------------
        # Stage 1: Detection & Analysis
        # ------------------------------------------------------------------
        print("\n[1/5] Detection & Analysis...")

        detection_start = datetime.now()

        detection_results = await self.detector.analyze_claim(
            claim,
            topic=topic,
            source_type=source_type,
        )

        # Entities (can be used by verification / impact later)
        claim.entities = await self.detector.extract_entities(claim_text)

        # Viral potential (0–1)
        claim.viral_score = await self.scanner.detect_viral_patterns(claim_text)

        # Sentiment (TextBlob)
        blob = TextBlob(claim_text)
        claim.sentiment_score = blob.sentiment.polarity

        print(f"  → Suspicion: {detection_results.get('suspicion_level', 'unknown')}")
        print(f"  → Viral Score: {claim.viral_score:.2f}")
        print(f"  → Entities: {', '.join(claim.entities[:3]) if claim.entities else 'None'}")

        # Store detection in context for downstream use / UI
        claim.context["detection"] = detection_results

        # ------------------------------------------------------------------
        # Stage 2: Verification
        # ------------------------------------------------------------------
        print("\n[2/5] Verification...")
        claim.status = ClaimStatus.VERIFYING

        verify_start = datetime.now()

        verification = await self.verifier.verify_claim(
            claim,
            detection=detection_results,
            topic=topic,
            source_type=source_type,
        )

        verify_end = datetime.now()
        verify_time = (verify_end - verify_start).total_seconds()

        # Update claim fields from verification
        claim.status = verification.status
        claim.confidence = verification.confidence
        claim.verification_summary = verification.summary
        claim.context.update(verification.context or {})
        claim.verified_at = datetime.now()

        print(f"  → Status: {verification.status.value}")
        print(f"  → Confidence: {int(verification.confidence * 100)}%")
        print(f"  → Verification time: {verify_time:.2f}s")

        # Update rolling average verification time
        prev_avg = self.stats.get("avg_verify_time", 0.0)
        prev_count = max(self.stats.get("claims_verified", 0), 1)
        self.stats["avg_verify_time"] = (prev_avg * prev_count + verify_time) / (prev_count + 1)

        # ------------------------------------------------------------------
        # Stage 3: Impact Analysis
        # ------------------------------------------------------------------
        print("\n[3/5] Impact Analysis...")

        impact = await self.analyzer.assess_impact(
            claim,
            verification,
            detection=detection_results,
            topic=topic,
            source_type=source_type,
        )

        severity_label = impact.get("severity", "medium")
        try:
            claim.severity = SeverityLevel(severity_label)
        except ValueError:
            claim.severity = SeverityLevel.MEDIUM

        harm_potential = impact.get("harm_potential", 0.5)

        print(f"  → Severity: {claim.severity.value}")
        print(f"  → Harm Potential: {harm_potential:.2f}")

        # Store impact in context
        claim.context["impact"] = impact

        # ------------------------------------------------------------------
        # Stage 4: Generate Communication
        # ------------------------------------------------------------------
        print("\n[4/5] Generating Public Alert...")

        public_alert = await self.communicator.generate_public_alert(
            claim,
            verification,
            impact=impact,
            audience=audience,
        )
        claim.context["public_alert"] = public_alert

        # ------------------------------------------------------------------
        # Stage 5: Store and Track
        # ------------------------------------------------------------------
        print("\n[5/5] Storing Results...")

        self.claims[claim_id] = claim
        self.claim_history.append(
            {
                "claim_id": claim_id,
                "timestamp": claim.detected_at.isoformat(),
                "status": claim.status.value,
                "confidence": claim.confidence,
                "severity": claim.severity.value,
            }
        )

        self.stats["claims_scanned"] += 1
        self.stats["claims_verified"] += 1
        if claim.status == ClaimStatus.FALSE:
            self.stats["misinformation_detected"] += 1

        print(f"\n✓ Processing Complete: {claim_id}")
        print(f"{'=' * 60}\n")

        return claim

    async def continuous_monitoring(self):
        """
        Background task: periodically scan configured sources and
        auto-process newly discovered claims.
        """
        while True:
            try:
                print("\n[Monitor] Scanning sources for new claims...")
                # You can adjust which sources to monitor and with what topics
                for source_type in getattr(self.scanner, "monitored_sources", [])[:3]:
                    # Here we don't pass a specific topic → generic stream.
                    # You could wire a per-source default topic map if you want.
                    claims = await self.scanner.scan_sources(source_type, topic=self.default_topic)

                    for t in claims:
                        cid = self.generate_claim_id(t)
                        if cid not in self.claims:
                            print(f"[Monitor] New claim detected from {source_type}: {t[:60]}...")
                            await self.process_claim(
                                t,
                                topic=self.default_topic,
                                source_type=source_type,
                                audience=self.default_audience,
                            )

                # Wait before next scan
                await asyncio.sleep(30)

            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(10)


# ============================================================================
# API ENDPOINTS
# ============================================================================

app = FastAPI(title="TruthGuard API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator: Optional[AgentOrchestrator] = None


class ClaimSubmission(BaseModel):
    text: str


class InitRequest(BaseModel):
    api_key: Optional[str] = None


@app.post("/api/init")
async def initialize_system(request: InitRequest):
    global orchestrator
    try:
        orchestrator = AgentOrchestrator(request.api_key)
        return {
            "status": "initialized",
            "message": "System ready",
            "provider": orchestrator.ai_provider.provider,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/claims/submit")
async def submit_claim(submission: ClaimSubmission, background_tasks: BackgroundTasks):
    if not orchestrator:
        raise HTTPException(status_code=400, detail="System not initialized")

    claim_id = orchestrator.generate_claim_id(submission.text)
    background_tasks.add_task(orchestrator.process_claim, submission.text)

    return {
        "claim_id": claim_id,
        "status": "processing",
        "message": "Claim submitted for verification",
    }


@app.get("/api/claims/{claim_id}")
async def get_claim(claim_id: str):
    if not orchestrator:
        raise HTTPException(status_code=400, detail="System not initialized")

    claim = orchestrator.claims.get(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    return {
        "id": claim.id,
        "text": claim.text,
        "status": claim.status.value,
        "confidence": claim.confidence,
        "severity": claim.severity.value,
        "viral_score": claim.viral_score,
        "detected_at": claim.detected_at.isoformat(),
        "verified_at": claim.verified_at.isoformat() if claim.verified_at else None,
        "summary": claim.verification_summary,
        "entities": claim.entities,
        "context": claim.context,
        "public_alert": claim.context.get("public_alert", ""),
    }


@app.get("/api/claims")
async def list_claims(limit: int = 20, status: Optional[str] = None):
    if not orchestrator:
        raise HTTPException(status_code=400, detail="System not initialized")

    claims = list(orchestrator.claims.values())

    if status:
        claims = [c for c in claims if c.status.value == status]

    claims.sort(key=lambda x: x.detected_at, reverse=True)
    claims = claims[:limit]

    return {
        "claims": [
            {
                "id": c.id,
                "text": c.text,
                "status": c.status.value,
                "confidence": c.confidence,
                "severity": c.severity.value,
                "detected_at": c.detected_at.isoformat(),
                "sources_count": len(c.sources),
            }
            for c in claims
        ],
        "total": len(orchestrator.claims),
    }


@app.get("/api/stats")
async def get_stats():
    if not orchestrator:
        raise HTTPException(status_code=400, detail="System not initialized")

    return orchestrator.stats


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": orchestrator is not None,
        "ai_provider": AI_PROVIDER,
    }


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            if orchestrator:
                recent_claims = list(orchestrator.claims.values())[-5:]
                await websocket.send_json(
                    {
                        "type": "update",
                        "claims": [
                            {
                                "id": c.id,
                                "text": c.text[:100],
                                "status": c.status.value,
                                "confidence": c.confidence,
                            }
                            for c in recent_claims
                        ],
                        "stats": orchestrator.stats,
                    }
                )

            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(
        f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                 🛡️  TRUTHGUARD BACKEND 2.0                   ║
    ║                                                              ║
    ║          Advanced AI Misinformation Detection System         ║
    ║                    AI Provider: {AI_PROVIDER or 'Not Detected':^21}          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Features:
    ✓ Multi-Agent AI Architecture
    ✓ Multi-Provider Support (OpenAI/Gemini/Claude)
    ✓ Real-time Web Search Integration
    ✓ Autonomous Claim Processing
    ✓ Advanced Impact Analysis
    ✓ Public Communication Generation
    
    Supported Providers:
    • OpenAI (ChatGPT)  - set OPENAI_API_KEY
    • Google Gemini     - set GOOGLE_API_KEY
    • Anthropic Claude  - set ANTHROPIC_API_KEY
    
    API Documentation: http://localhost:8000/docs
    
    """
    )

    if not AI_PROVIDER:
        print("⚠️  WARNING: No AI provider detected!")
        print("   Check your .env or environment variables and libraries.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
