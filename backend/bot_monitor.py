"""
Bot Monitoring Service
Runs continuously to monitor social media for bot mentions
"""
import time
import requests
import schedule
from agents.orchestrator_agent import OrchestratorAgent
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BotMonitor:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.api_url = "http://localhost:5000"
    
    def check_mentions(self):
        """Check and process bot mentions"""
        logger.info("Checking for bot mentions...")
        
        try:
            results = self.orchestrator.process_bot_mentions_batch()
            
            if results:
                logger.info(f"Processed {len(results)} mentions")
                
                for result in results:
                    if result.get('reply_posted'):
                        logger.info(f"✅ Reply posted for: {result.get('claim', '')[:50]}")
                    elif 'error' in result:
                        logger.error(f"❌ Error: {result['error']}")
            else:
                logger.info("No new mentions found")
        
        except Exception as e:
            logger.error(f"Error checking mentions: {e}")
    
    def crawl_sources(self):
        """Periodic crawling of trusted sources"""
        logger.info("Crawling trusted sources...")
        
        try:
            count = self.orchestrator.crawl_and_index_sources()
            logger.info(f"✅ Indexed {count} new articles")
        except Exception as e:
            logger.error(f"Error crawling sources: {e}")
    
    def monitor_trending(self):
        """Monitor for trending crisis claims"""
        logger.info("Monitoring trending claims...")
        
        keywords = ['tsunami', 'earthquake', 'flood', 'pandemic', 'outbreak', 'warning']
        
        try:
            trending = self.orchestrator.monitor_trending_claims(keywords)
            
            if trending:
                logger.info(f"Found {len(trending)} trending claims")
                
                for claim in trending:
                    logger.info(f"📊 Trending: {claim.get('claim', '')[:60]} - {claim.get('verdict')}")
            else:
                logger.info("No trending claims detected")
        
        except Exception as e:
            logger.error(f"Error monitoring trending: {e}")
    
    def start(self):
        """Start the monitoring service"""
        logger.info("🤖 Starting VeriPulse Bot Monitor...")
        
        # Schedule tasks
        schedule.every(2).minutes.do(self.check_mentions)  # Check mentions every 2 minutes
        schedule.every(30).minutes.do(self.crawl_sources)  # Crawl sources every 30 minutes
        schedule.every(15).minutes.do(self.monitor_trending)  # Check trending every 15 minutes
        
        # Run initial tasks
        self.crawl_sources()
        self.check_mentions()
        
        # Keep running
        logger.info("✅ Bot monitor started successfully")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Bot monitor stopped")

if __name__ == "__main__":
    monitor = BotMonitor()
    monitor.start()