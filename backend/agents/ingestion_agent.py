# backend/agents/ingestion_agent.py
import logging
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    IngestionAgent handles fetching content from social sources (Twitter/Reddit/RSS)
    and converting them into articles for the pipeline.
    """

    def __init__(self, use_simulation: bool = False, poll_interval: int = 60):
        """
        Args:
            use_simulation: if True, do not attempt to initialize real social API clients.
            poll_interval: seconds between polling loops when start_polling is used.
        """
        self.use_simulation = bool(use_simulation)
        self.poll_interval = int(poll_interval)
        self.simulated_data_counter = 0

        # Determine whether we have any social creds from Config
        twitter_creds = any([
            Config.TWITTER_BEARER_TOKEN,
            Config.TWITTER_API_KEY and Config.TWITTER_API_SECRET and Config.TWITTER_ACCESS_TOKEN and Config.TWITTER_ACCESS_SECRET
        ])
        reddit_creds = bool(Config.REDDIT_CLIENT_ID and Config.REDDIT_CLIENT_SECRET)

        # Final decision: if use_simulation True OR no credentials -> run simulation
        if self.use_simulation:
            logger.info("IngestionAgent starting in SIMULATION mode (forced by parameter).")
            self._init_simulation()
        elif not (twitter_creds or reddit_creds):
            logger.info("No social credentials found — IngestionAgent starting in SIMULATION mode.")
            self.use_simulation = True
            self._init_simulation()
        else:
            # Attempt to initialize real API clients; if anything fails, fallback to simulation
            try:
                logger.info("Initializing real social API clients for ingestion.")
                self._init_social_clients()
            except Exception as e:
                logger.exception(f"Failed to initialize social API clients; falling back to simulation. Error: {e}")
                self.use_simulation = True
                self._init_simulation()

    # -------------------------
    # Initialization helpers
    # -------------------------
    def _init_social_clients(self):
        """Initialize real API clients (Twitter, Reddit)."""
        try:
            import tweepy
        except Exception:
            tweepy = None

        try:
            import praw
        except Exception:
            praw = None

        self.twitter_client = None
        self.reddit_client = None

        if Config.TWITTER_BEARER_TOKEN or (Config.TWITTER_API_KEY and Config.TWITTER_API_SECRET):
            if tweepy is None:
                logger.warning("tweepy not installed — Twitter ingestion disabled.")
            else:
                logger.info("Initializing tweepy client.")
                if Config.TWITTER_BEARER_TOKEN:
                    client = tweepy.Client(bearer_token=Config.TWITTER_BEARER_TOKEN)
                    self.twitter_client = client
                else:
                    auth = tweepy.OAuth1UserHandler(Config.TWITTER_API_KEY, Config.TWITTER_API_SECRET,
                                                    Config.TWITTER_ACCESS_TOKEN, Config.TWITTER_ACCESS_SECRET)
                    api = tweepy.API(auth)
                    self.twitter_client = api

        if Config.REDDIT_CLIENT_ID and Config.REDDIT_CLIENT_SECRET:
            if praw is None:
                logger.warning("praw not installed — Reddit ingestion disabled.")
            else:
                logger.info("Initializing praw Reddit client.")
                self.reddit_client = praw.Reddit(
                    client_id=Config.REDDIT_CLIENT_ID,
                    client_secret=Config.REDDIT_CLIENT_SECRET,
                    user_agent=Config.REDDIT_USER_AGENT
                )

        if not self.twitter_client and not self.reddit_client:
            raise RuntimeError("No social clients could be initialized (missing libs or credentials).")

    def _init_simulation(self):
        """Prepare simulated ingestion mode."""
        self.twitter_client = None
        self.reddit_client = None
        logger.info("IngestionAgent: running with simulated ingestion. Will generate mock items on demand.")

    # -------------------------
    # Public ingestion methods
    # -------------------------
    def ingest_sources(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ingest content from trusted sources or social platforms.
        Returns list of 'article' dicts with keys: title, text, url, timestamp.
        """
        logger.debug(f"IngestionAgent.ingest_sources called (limit={limit}) use_simulation={self.use_simulation}")
        if self.use_simulation:
            return self._simulate_ingest(limit)
        
        results = []
        try:
            import feedparser
            for feed_url in Config.RSS_FEEDS:
                d = feedparser.parse(feed_url)
                for entry in d.entries[:limit]:
                    results.append({
                        "title": entry.get("title", "")[:200],
                        "text": entry.get("summary", "")[:4000],
                        "url": entry.get("link", ""),
                        "timestamp": entry.get("published", None)
                    })
                    if len(results) >= limit:
                        break
                if len(results) >= limit:
                    break
        except Exception as e:
            logger.exception(f"RSS ingestion failed: {e}")
        return results

    def stream_crisis_keywords(self, keywords: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """
        Stream/search social media for posts matching crisis keywords.
        Returns list of posts with text, author, platform, timestamp.
        """
        logger.debug(f"stream_crisis_keywords called with keywords={keywords}, limit={limit}")
        
        if self.use_simulation:
            return self._simulate_crisis_posts(keywords, limit)
        
        posts = []
        
        # Search Twitter
        if self.twitter_client:
            try:
                query = " OR ".join(keywords)
                tweets = self.twitter_client.search_recent_tweets(
                    query=query, 
                    max_results=min(limit, 100),
                    tweet_fields=['created_at', 'author_id']
                )
                
                if tweets and tweets.data:
                    for tweet in tweets.data:
                        posts.append({
                            "id": str(tweet.id),
                            "text": tweet.text,
                            "author_id": str(tweet.author_id) if hasattr(tweet, 'author_id') else "unknown",
                            "platform": "twitter",
                            "timestamp": str(tweet.created_at) if hasattr(tweet, 'created_at') else None,
                            "url": f"https://twitter.com/i/web/status/{tweet.id}"
                        })
            except Exception as e:
                logger.exception(f"Error searching Twitter: {e}")
        
        # Search Reddit
        if self.reddit_client:
            try:
                for keyword in keywords[:3]:  # Limit to avoid rate limits
                    subreddit = self.reddit_client.subreddit("all")
                    for submission in subreddit.search(keyword, limit=min(limit // len(keywords), 25)):
                        posts.append({
                            "id": str(submission.id),
                            "text": f"{submission.title}\n{submission.selftext or ''}",
                            "author_id": str(submission.author) if submission.author else "unknown",
                            "platform": "reddit",
                            "timestamp": str(submission.created_utc),
                            "url": f"https://reddit.com{submission.permalink}"
                        })
            except Exception as e:
                logger.exception(f"Error searching Reddit: {e}")
        
        return posts[:limit]

    def collect_twitter_mentions(self) -> List[Dict[str, Any]]:
        """
        Collect recent Twitter mentions of the bot.
        Returns list of mention dicts.
        """
        logger.debug("collect_twitter_mentions called")
        
        if self.use_simulation:
            return self._simulate_mentions(5)
        
        mentions = []
        if self.twitter_client:
            try:
                # Search for mentions (adjust query for your bot's handle)
                tweets = self.twitter_client.search_recent_tweets(
                    query="@VeriPulseBot OR #VeriCheck",
                    max_results=20,
                    tweet_fields=['created_at', 'author_id']
                )
                
                if tweets and tweets.data:
                    for tweet in tweets.data:
                        mentions.append({
                            "platform": "twitter",
                            "post_id": str(tweet.id),
                            "author_id": str(tweet.author_id) if hasattr(tweet, 'author_id') else "unknown",
                            "text": tweet.text,
                            "timestamp": str(tweet.created_at) if hasattr(tweet, 'created_at') else None
                        })
            except Exception as e:
                logger.exception(f"Error collecting Twitter mentions: {e}")
        
        return mentions

    def collect_reddit_mentions(self) -> List[Dict[str, Any]]:
        """
        Collect recent Reddit mentions of the bot.
        Returns list of mention dicts.
        """
        logger.debug("collect_reddit_mentions called")
        
        if self.use_simulation:
            return self._simulate_mentions(5)
        
        mentions = []
        if self.reddit_client:
            try:
                # Search for mentions (adjust for your bot's username)
                for mention in self.reddit_client.inbox.mentions(limit=20):
                    mentions.append({
                        "platform": "reddit",
                        "post_id": str(mention.id),
                        "author_id": str(mention.author) if mention.author else "unknown",
                        "text": mention.body,
                        "timestamp": str(mention.created_utc)
                    })
            except Exception as e:
                logger.exception(f"Error collecting Reddit mentions: {e}")
        
        return mentions

    def process_mentions_batch(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Process recent mentions / posts for downstream agents.
        Returns a list of processed mention dicts.
        """
        logger.debug(f"IngestionAgent.process_mentions_batch called (limit={limit})")
        
        mentions = []
        mentions.extend(self.collect_twitter_mentions())
        mentions.extend(self.collect_reddit_mentions())
        
        return mentions[:limit]

    # -------------------------
    # Simulation helpers
    # -------------------------
    def _simulate_ingest(self, limit: int = 10) -> List[Dict[str, Any]]:
        items = []
        for i in range(limit):
            self.simulated_data_counter += 1
            items.append({
                "title": f"Simulated Article #{self.simulated_data_counter}",
                "text": f"This is simulated content to test ingestion pipeline. Item #{self.simulated_data_counter}.",
                "url": f"https://example.com/simulated/{self.simulated_data_counter}",
                "timestamp": None
            })
        logger.debug(f"Generated {len(items)} simulated articles.")
        return items

    def _simulate_crisis_posts(self, keywords: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """Simulate crisis-related social media posts"""
        posts = []
        crisis_templates = [
            "Breaking: {keyword} reported in multiple areas. Stay safe!",
            "Urgent: {keyword} alert issued for the region. Follow official guidance.",
            "Is the {keyword} situation getting worse? Need verification.",
            "Local authorities confirm {keyword} incident. More details needed.",
            "Rumors about {keyword} spreading fast. Can anyone verify?",
        ]
        
        for i in range(limit):
            self.simulated_data_counter += 1
            keyword = keywords[i % len(keywords)]
            template = crisis_templates[i % len(crisis_templates)]
            
            posts.append({
                "id": f"sim-crisis-{self.simulated_data_counter}",
                "text": template.format(keyword=keyword),
                "author_id": f"sim_user_{i % 10}",
                "platform": "simulation",
                "timestamp": None,
                "url": f"https://example.com/post/{self.simulated_data_counter}"
            })
        
        logger.debug(f"Generated {len(posts)} simulated crisis posts for keywords: {keywords}")
        return posts

    def _simulate_mentions(self, limit: int = 20) -> List[Dict[str, Any]]:
        items = []
        for i in range(limit):
            self.simulated_data_counter += 1
            items.append({
                "platform": "simulation",
                "post_id": f"sim-{self.simulated_data_counter}",
                "author_id": "sim_user",
                "text": f"@veripulse Is this true? Simulated mention #{self.simulated_data_counter}",
                "timestamp": None
            })
        logger.debug(f"Generated {len(items)} simulated mentions.")
        return items

    # -------------------------
    # Utility / lifecycle
    # -------------------------
    def start_polling(self):
        """
        Start a non-blocking polling loop to periodically ingest sources.
        This is a non-blocking stub: actual scheduling should be done outside (Celery / APScheduler).
        """
        logger.info("start_polling called — this is a no-op stub in the lightweight agent. Implement scheduler externally.")

    def shutdown(self):
        """Clean up clients if necessary."""
        logger.info("Shutting down IngestionAgent.")
        try:
            if getattr(self, "twitter_client", None) and hasattr(self.twitter_client, "close"):
                self.twitter_client.close()
        except Exception:
            pass
        try:
            if getattr(self, "reddit_client", None) and hasattr(self.reddit_client, "close"):
                self.reddit_client.close()
        except Exception:
            pass