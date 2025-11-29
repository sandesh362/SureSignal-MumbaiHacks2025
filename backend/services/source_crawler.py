import requests
from bs4 import BeautifulSoup
import feedparser
from newspaper import Article
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
from config import Config
import logging

logger = logging.getLogger(__name__)

class SourceCrawler:
    def __init__(self):
        self.trusted_sources = Config.TRUSTED_SOURCES
        self.rss_feeds = Config.RSS_FEEDS
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl_rss_feeds(self) -> List[Dict]:
        """Crawl all RSS feeds for latest articles"""
        articles = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:
                    article_data = {
                        "title": entry.get('title', ''),
                        "url": entry.get('link', ''),
                        "published": entry.get('published', ''),
                        "summary": entry.get('summary', ''),
                        "source": self._identify_source(feed_url),
                        "crawled_at": datetime.utcnow().isoformat()
                    }
                    
                    article_data['id'] = self._generate_id(article_data['url'])
                    articles.append(article_data)
                    
            except Exception as e:
                logger.warning(f"Error crawling RSS feed {feed_url}: {e}")
        
        return articles
    
    def fetch_article_content(self, url: str) -> Optional[Dict]:
        """Fetch full article content from URL"""
        try:
            # FIXED: Validate URL before processing
            if not url or not url.startswith(('http://', 'https://')):
                logger.warning(f"Invalid URL format: {url}")
                return None
            
            # Skip non-article URLs
            if self._is_excluded_url(url):
                logger.debug(f"Skipping excluded URL: {url}")
                return None
            
            article = Article(url)
            article.download()
            article.parse()
            
            return {
                "url": url,
                "title": article.title,
                "text": article.text,
                "authors": article.authors,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "top_image": article.top_image,
                "source": self._identify_source(url),
                "crawled_at": datetime.utcnow().isoformat(),
                "id": self._generate_id(url)
            }
        except Exception as e:
            logger.warning(f"Error fetching article {url}: {e}")
            return None
    
    def search_trusted_source(self, query: str, source_name: str) -> List[Dict]:
        """Search a specific trusted source for relevant content"""
        source_url = self.trusted_sources.get(source_name.lower())
        
        if not source_url:
            return []
        
        try:
            search_url = f"{source_url}/search?q={query}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            for link in soup.find_all('a', href=True)[:5]:
                href = link['href']
                
                # FIXED: Better URL normalization
                if href.startswith('/'):
                    href = source_url.rstrip('/') + href
                elif not href.startswith(('http://', 'https://')):
                    continue  # Skip invalid URLs
                
                if self._is_article_url(href):
                    article = self.fetch_article_content(href)
                    if article:
                        articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.warning(f"Error searching {source_name}: {e}")
            return []
    
    def crawl_pib_latest(self) -> List[Dict]:
        """Crawl latest PIB releases"""
        try:
            url = "https://pib.gov.in/allRel.aspx"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            for item in soup.select('.content-area')[:10]:
                title_elem = item.select_one('h3, .title')
                link_elem = item.select_one('a')
                
                if title_elem and link_elem:
                    article_url = link_elem.get('href', '')
                    
                    # FIXED: Validate URL
                    if not article_url or article_url.startswith(('#', 'javascript:')):
                        continue
                    
                    if not article_url.startswith('http'):
                        article_url = 'https://pib.gov.in' + article_url
                    
                    articles.append({
                        "title": title_elem.get_text(strip=True),
                        "url": article_url,
                        "source": "pib",
                        "crawled_at": datetime.utcnow().isoformat(),
                        "id": self._generate_id(article_url)
                    })
            
            return articles
            
        except Exception as e:
            logger.warning(f"Error crawling PIB: {e}")
            return []
    
    def crawl_who_latest(self) -> List[Dict]:
        """Crawl latest WHO news"""
        try:
            url = "https://www.who.int/news"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            for item in soup.select('.list-view--item')[:10]:
                title_elem = item.select_one('.link-container a')
                
                if title_elem:
                    article_url = title_elem.get('href', '')
                    
                    # FIXED: Validate URL
                    if not article_url or article_url.startswith(('#', 'javascript:')):
                        continue
                    
                    if not article_url.startswith('http'):
                        article_url = 'https://www.who.int' + article_url
                    
                    articles.append({
                        "title": title_elem.get_text(strip=True),
                        "url": article_url,
                        "source": "who",
                        "crawled_at": datetime.utcnow().isoformat(),
                        "id": self._generate_id(article_url)
                    })
            
            return articles
            
        except Exception as e:
            logger.warning(f"Error crawling WHO: {e}")
            return []
    
    def crawl_all_sources(self) -> List[Dict]:
        """Crawl all trusted sources"""
        all_articles = []
        
        all_articles.extend(self.crawl_rss_feeds())
        all_articles.extend(self.crawl_pib_latest())
        all_articles.extend(self.crawl_who_latest())
        
        return all_articles
    
    def _identify_source(self, url: str) -> str:
        """Identify source from URL"""
        url_lower = url.lower()
        
        for source_name, source_url in self.trusted_sources.items():
            source_domain = source_url.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
            if source_domain in url_lower:
                return source_name
        
        return "unknown"
    
    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article"""
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        excluded = ['login', 'signup', 'subscribe', 'privacy', 'about', 'contact', 
                   'search', 'javascript:', '#']
        return not any(ex in url.lower() for ex in excluded)
    
    def _is_excluded_url(self, url: str) -> bool:
        """Check if URL should be excluded from crawling"""
        excluded_patterns = [
            'index.aspx', '#content', 'javascript:', '#', 
            '/search', '/login', '/signup', 'mailto:'
        ]
        return any(pattern in url.lower() for pattern in excluded_patterns)
    
    def _generate_id(self, url: str) -> str:
        """Generate unique ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()