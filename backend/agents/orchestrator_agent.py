# backend/agents/orchestrator_agent.py
from crewai import Agent, Crew, Task
from langchain_openai import ChatOpenAI
from typing import Dict, List
import logging

# Import local agents & services
from agents.ingestion_agent import IngestionAgent
from agents.claim_extraction_agent import ClaimExtractionAgent
from agents.evidence_retrieval_agent import EvidenceRetrievalAgent
from agents.veracity_agent import VeracityAgent
from agents.explanation_agent import ExplanationAgent

from services.mongodb_service import MongoDBService
from services.vector_store import VectorStoreService
from services.source_crawler import SourceCrawler
from services.llm_service import LLMService

from config import Config

logger = logging.getLogger(__name__)

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except Exception:
    TWEEPY_AVAILABLE = False

try:
    import praw
    PRAW_AVAILABLE = True
except Exception:
    PRAW_AVAILABLE = False


class OrchestratorAgent:
    def __init__(self):
        llm_service = LLMService()
        self.llm = llm_service.get_chat_llm(
            model="llama-local",
            temperature=0.2,
            system_prompt="You coordinate fact verification agents."
        )

        # Determine whether social integrations are enabled from Config
        self.twitter_enabled = bool(
            getattr(Config, "TWITTER_BEARER_TOKEN", None) or
            getattr(Config, "TWITTER_API_KEY", None)
        ) and TWEEPY_AVAILABLE

        self.reddit_enabled = bool(
            getattr(Config, "REDDIT_CLIENT_ID", None)
        ) and PRAW_AVAILABLE

        # Initialize services
        self.mongo_service = MongoDBService()
        self.vector_store = VectorStoreService()
        self.crawler = SourceCrawler()

        # Initialize all agents
        # ingestion runs in simulation mode when social creds are missing
        self.ingestion_agent = IngestionAgent(use_simulation=not self._has_social_creds())
        self.claim_agent = ClaimExtractionAgent()
        self.evidence_agent = EvidenceRetrievalAgent(
            self.vector_store,
            self.crawler,
            self.mongo_service
        )
        self.veracity_agent = VeracityAgent()
        self.explanation_agent = ExplanationAgent()

        # Initialize bot clients (may return None if not enabled)
        self.twitter_client = self._init_twitter_bot() if self.twitter_enabled else None
        self.reddit_client = self._init_reddit_bot() if self.reddit_enabled else None

        # Agent wrapper for coordination
        self.agent = Agent(
            role="Verification Orchestrator",
            goal="Coordinate all agents to verify claims efficiently",
            backstory="""You are the master coordinator who orchestrates 
            the entire verification pipeline. You ensure smooth flow between 
            agents, handle rate limiting, and deliver results.""",
            llm=self.llm,
            verbose=True
        )

    def _has_social_creds(self) -> bool:
        """Check if any social media credentials are configured"""
        return bool(
            getattr(Config, "TWITTER_BEARER_TOKEN", None) or
            getattr(Config, "REDDIT_CLIENT_ID", None)
        )

    def _init_twitter_bot(self):
        """Initialize Twitter bot for posting replies (only if enabled and tweepy available)"""
        if not TWEEPY_AVAILABLE:
            logger.info("Tweepy not available; skipping Twitter client initialization.")
            return None
        try:
            if getattr(Config, "TWITTER_API_KEY", None):
                return tweepy.Client(
                    bearer_token=Config.TWITTER_BEARER_TOKEN,
                    consumer_key=Config.TWITTER_API_KEY,
                    consumer_secret=Config.TWITTER_API_SECRET,
                    access_token=Config.TWITTER_ACCESS_TOKEN,
                    access_token_secret=Config.TWITTER_ACCESS_SECRET
                )
            logger.info("Twitter config missing API key; Twitter client not initialized.")
        except Exception as e:
            logger.warning(f"Failed to init Twitter bot: {e}")
        return None

    def _init_reddit_bot(self):
        """Initialize Reddit bot for posting replies (only if enabled and praw available)"""
        if not PRAW_AVAILABLE:
            logger.info("PRAW not available; skipping Reddit client initialization.")
            return None
        try:
            if getattr(Config, "REDDIT_CLIENT_ID", None):
                return praw.Reddit(
                    client_id=Config.REDDIT_CLIENT_ID,
                    client_secret=Config.REDDIT_CLIENT_SECRET,
                    user_agent=Config.REDDIT_USER_AGENT
                )
            logger.info("Reddit config missing client id; Reddit client not initialized.")
        except Exception as e:
            logger.warning(f"Failed to init Reddit bot: {e}")
        return None

    def verify_single_claim(self, text: str, user_id: str = None,
                           platform: str = "web") -> Dict:
        """Verify a single claim (from bot tag or web portal)"""

        # Check rate limit
        if user_id and platform != "web":
            rate_count = self.mongo_service.get_user_rate_limit(user_id)
            if rate_count >= Config.RATE_LIMIT_REQUESTS:
                return {
                    "error": "Rate limit exceeded",
                    "message": "You've reached the maximum requests per hour. Please try again later."
                }

        # Step 1: Store claim
        claim_data = {
            "claim_text": text,
            "platform": platform,
            "user_id": user_id
        }

        claim_id = self.mongo_service.store_claim(claim_data)
        claim_data['id'] = claim_id

        # Step 2: Retrieve evidence
        try:
            evidence_result = self.evidence_agent.retrieve_evidence(claim_data)
        except Exception as e:
            logger.exception(f"Evidence retrieval failed: {e}")
            evidence_result = {
                "evidence": [],
                "error": str(e)
            }

        # Step 3: Verify
        try:
            verification = self.veracity_agent.verify_claim(
                text,
                evidence_result.get('evidence', [])
            )
        except Exception as e:
            logger.exception(f"Verification failed: {e}")
            verification = {
                "verdict": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Verification error: {str(e)}",
                "evidence_used": []
            }

        # Step 4: Explain
        try:
            explanation = self.explanation_agent.generate_explanation(verification)
        except Exception as e:
            logger.exception(f"Explanation generation failed: {e}")
            explanation = {
                "summary": f"{verification.get('verdict', 'ERROR')}",
                "detailed_explanation": verification.get('reasoning', 'Error generating explanation'),
                "eli12_explanation": "We couldn't verify this claim due to an error.",
                "citations": []
            }

        # Store verification
        verification_data = {
            "claim_id": claim_id,
            "verdict": verification.get('verdict', 'ERROR'),
            "confidence": verification.get('confidence', 0.0),
            "reasoning": verification.get('reasoning', ''),
            "evidence_count": len(evidence_result.get('evidence', []))
        }
        self.mongo_service.store_verification(verification_data)

        # Store bot interaction
        if user_id:
            self.mongo_service.store_bot_interaction({
                "user_id": user_id,
                "platform": platform,
                "claim_id": claim_id,
                "verdict": verification.get('verdict', 'ERROR')
            })

        return {
            "claim": text,
            "verdict": verification.get('verdict', 'ERROR'),
            "confidence": verification.get('confidence', 0.0),
            "explanation": explanation,
            "evidence": evidence_result.get('evidence', [])
        }

    def process_bot_mention(self, mention: Dict) -> Dict:
        """Process a bot mention from Twitter/Reddit"""
        platform = mention['platform']
        post_id = mention['post_id']
        author_id = mention['author_id']
        text = mention['text']

        # Extract claim from text (remove bot mention)
        claim_text = text.replace("@VeriPulseBot", "").replace("#VeriCheck", "").strip()

        # Verify the claim
        result = self.verify_single_claim(claim_text, author_id, platform)

        if "error" in result:
            return result

        # Format reply
        try:
            reply_text = self.explanation_agent.format_bot_reply(
                claim_text,
                result['explanation']
            )
        except Exception as e:
            logger.exception(f"Error formatting reply: {e}")
            reply_text = f"Verdict: {result['verdict']} ({result['confidence']*100:.0f}% confidence)"

        # Post reply only if the corresponding client is enabled and initialized
        if platform == "twitter":
            if self.twitter_enabled and self.twitter_client:
                try:
                    self.twitter_client.create_tweet(
                        text=reply_text,
                        in_reply_to_tweet_id=post_id
                    )
                    result['reply_posted'] = True
                except Exception as e:
                    logger.exception(f"Error posting Twitter reply: {e}")
                    result['reply_error'] = str(e)
            else:
                logger.info("Twitter posting skipped: Twitter integration not enabled or client not initialized.")
                result['reply_posted'] = False
                result['reply_note'] = "Twitter disabled"

        elif platform == "reddit":
            if self.reddit_enabled and self.reddit_client:
                try:
                    # Using praw to comment - ensure correct API usage for your praw version
                    submission = self.reddit_client.submission(id=post_id)
                    submission.reply(reply_text)
                    result['reply_posted'] = True
                except Exception as e:
                    logger.exception(f"Error posting Reddit reply: {e}")
                    result['reply_error'] = str(e)
            else:
                logger.info("Reddit posting skipped: Reddit integration not enabled or client not initialized.")
                result['reply_posted'] = False
                result['reply_note'] = "Reddit disabled"

        return result

    def monitor_trending_claims(self, keywords: List[str]) -> List[Dict]:
        """Monitor social media for trending crisis claims"""

        try:
            # Collect posts using ingestion agent; ingestion agent may simulate when creds missing
            posts = self.ingestion_agent.stream_crisis_keywords(keywords, limit=100)

            if not posts:
                logger.info("No posts found for trending keywords")
                return []

            logger.info(f"Collected {len(posts)} posts for trending analysis")

            # Extract and cluster claims
            extraction_result = self.claim_agent.process_batch(posts)
            trending_claims = extraction_result.get('trending', [])

            if not trending_claims:
                logger.info("No trending claims extracted")
                return []

            logger.info(f"Extracted {len(trending_claims)} trending claims")

            # Verify top trending claims
            verified_claims = []

            for claim in trending_claims[:10]:  # Top 10 trending
                try:
                    claim_text = claim.get('claim_text', '')
                    if not claim_text:
                        continue

                    # Check if already verified recently
                    existing = self.mongo_service.get_verification_history(claim.get('id', ''))

                    if existing and len(existing) > 0:
                        # Use cached verification
                        verified_claims.append(existing[0])
                        logger.info(f"Using cached verification for: {claim_text[:50]}")
                    else:
                        # Verify new claim
                        logger.info(f"Verifying trending claim: {claim_text[:50]}")
                        result = self.verify_single_claim(
                            claim_text,
                            platform="trending"
                        )
                        verified_claims.append(result)

                except Exception as e:
                    logger.exception(f"Error verifying trending claim: {e}")
                    continue

            return verified_claims

        except Exception as e:
            logger.exception(f"Error in monitor_trending_claims: {e}")
            return []

    def process_bot_mentions_batch(self) -> List[Dict]:
        """Process all pending bot mentions"""
        results = []

        # Collect Twitter mentions only if twitter is enabled
        if self.twitter_enabled and self.twitter_client:
            try:
                twitter_mentions = self.ingestion_agent.collect_twitter_mentions()
                for mention in twitter_mentions:
                    result = self.process_bot_mention(mention)
                    results.append(result)
            except Exception as e:
                logger.exception(f"Error processing Twitter mentions: {e}")
        else:
            logger.info("Skipping Twitter mention processing: Twitter integration not enabled or client not initialized.")

        # Collect Reddit mentions only if reddit is enabled
        if self.reddit_enabled and self.reddit_client:
            try:
                reddit_mentions = self.ingestion_agent.collect_reddit_mentions()
                for mention in reddit_mentions:
                    result = self.process_bot_mention(mention)
                    results.append(result)
            except Exception as e:
                logger.exception(f"Error processing Reddit mentions: {e}")
        else:
            logger.info("Skipping Reddit mention processing: Reddit integration not enabled or client not initialized.")

        return results

    def crawl_and_index_sources(self):
        """Crawl trusted sources and index in vector DB"""
        try:
            articles = self.crawler.crawl_all_sources()

            batch = []
            for article in articles:
                # Store in MongoDB
                evidence_id = self.mongo_service.store_evidence(article)

                # Prepare for vector indexing
                text = f"{article.get('title', '')} {article.get('summary', '')}"

                batch.append({
                    "id": evidence_id,
                    "text": text,
                    "metadata": {
                        "source": article.get('source'),
                        "url": article.get('url'),
                        "title": article.get('title'),
                        "crawled_at": article.get('crawled_at')
                    }
                })

            # Batch index
            if batch:
                self.vector_store.upsert_evidence_batch(batch)

            return len(batch)
        except Exception as e:
            logger.exception(f"Error in crawl_and_index_sources: {e}")
            return 0

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        try:
            mongo_stats = self.mongo_service.get_statistics()
            vector_stats = self.vector_store.get_index_stats()

            return {
                "mongodb": mongo_stats,
                "vector_db": vector_stats,
                "agents_active": 6,
                "simulation_mode": self.ingestion_agent.use_simulation,
                "twitter_enabled": self.twitter_enabled,
                "reddit_enabled": self.reddit_enabled
            }
        except Exception as e:
            logger.exception(f"Error getting statistics: {e}")
            return {
                "error": str(e),
                "agents_active": 6,
                "simulation_mode": True
            }
