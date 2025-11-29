from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from typing import Dict, List, Optional
from services.vector_store import VectorStoreService
from services.source_crawler import SourceCrawler
from services.mongodb_service import MongoDBService
from config import Config
from services.llm_service import LLMService

class EvidenceRetrievalAgent:
    def __init__(self, vector_store: VectorStoreService, crawler: SourceCrawler, mongo_service: MongoDBService):
        self.vector_store = vector_store
        self.crawler = crawler
        self.mongo_service = mongo_service
        llm_service = LLMService()
        self.llm = llm_service.get_chat_llm(model="llama-local", temperature=0.3)
        
        self.agent = Agent(
            role="Evidence Retrieval Specialist",
            goal="Find relevant evidence from trusted sources to verify claims",
            backstory="""You are an expert researcher who knows how to find 
            authoritative sources quickly. You prioritize government sources, 
            official statements, and verified news outlets. You understand 
            context and can identify the most relevant evidence.""",
            llm=self.llm,
            verbose=True
        )
    
    def retrieve_evidence(self, claim: Dict) -> Dict:
        """Retrieve evidence for a claim using RAG"""
        claim_text = claim['claim_text']
        
        # 1. Search vector database for similar evidence
        vector_results = self.vector_store.search_similar_evidence(
            query_text=claim_text,
            top_k=Config.MAX_SOURCES_PER_CLAIM
        )
        
        # 2. If insufficient results, crawl sources in real-time
        if len(vector_results) < 3 or max([r['score'] for r in vector_results] or [0]) < 0.7:
            fresh_evidence = self._crawl_fresh_evidence(claim_text)
            
            # Store fresh evidence in vector DB
            if fresh_evidence:
                self._index_fresh_evidence(fresh_evidence)
                
                # Re-search with updated index
                vector_results = self.vector_store.search_similar_evidence(
                    query_text=claim_text,
                    top_k=Config.MAX_SOURCES_PER_CLAIM
                )
        
        # 3. Rank and filter evidence
        ranked_evidence = self._rank_evidence(vector_results, claim_text)
        
        return {
            "claim_id": claim.get('id'),
            "claim_text": claim_text,
            "evidence": ranked_evidence,
            "evidence_count": len(ranked_evidence),
            "max_similarity": max([e['relevance_score'] for e in ranked_evidence]) if ranked_evidence else 0
        }
    
    def _crawl_fresh_evidence(self, claim_text: str) -> List[Dict]:
        """Crawl trusted sources for fresh evidence"""
        evidence = []
        
        # Extract keywords for targeted crawling
        keywords = self._extract_keywords(claim_text)
        
        # Crawl PIB
        pib_articles = self.crawler.search_trusted_source(keywords, "pib")
        for article in pib_articles[:2]:
            if article:
                evidence.append({
                    "source": "pib",
                    "title": article.get('title'),
                    "text": article.get('text'),
                    "url": article.get('url'),
                    "published_date": article.get('publish_date')
                })
        
        # Crawl WHO
        who_articles = self.crawler.search_trusted_source(keywords, "who")
        for article in who_articles[:2]:
            if article:
                evidence.append({
                    "source": "who",
                    "title": article.get('title'),
                    "text": article.get('text'),
                    "url": article.get('url'),
                    "published_date": article.get('publish_date')
                })
        
        # Crawl AP News
        ap_articles = self.crawler.search_trusted_source(keywords, "ap_news")
        for article in ap_articles[:2]:
            if article:
                evidence.append({
                    "source": "ap_news",
                    "title": article.get('title'),
                    "text": article.get('text'),
                    "url": article.get('url'),
                    "published_date": article.get('publish_date')
                })
        
        return evidence
    
    def _index_fresh_evidence(self, evidence_list: List[Dict]):
        """Index newly crawled evidence in vector database"""
        batch = []
        
        for evidence in evidence_list:
            # Store in MongoDB
            evidence_id = self.mongo_service.store_evidence(evidence)
            
            # Prepare for vector indexing
            text_content = f"{evidence.get('title', '')} {evidence.get('text', '')}"
            
            batch.append({
                "id": evidence_id,
                "text": text_content,
                "metadata": {
                    "source": evidence.get('source'),
                    "url": evidence.get('url'),
                    "title": evidence.get('title'),
                    "published_date": evidence.get('published_date'),
                    "text": evidence.get('text', '')[:500]  # Store snippet in metadata
                }
            })
        
        if batch:
            self.vector_store.upsert_evidence_batch(batch)
    
    def _rank_evidence(self, vector_results: List[Dict], claim_text: str) -> List[Dict]:
        """Rank evidence by relevance and source credibility"""
        ranked = []
        
        source_weights = {
            "pib": 1.0,
            "who": 1.0,
            "imd": 1.0,
            "ndma": 1.0,
            "reuters": 0.9,
            "ap_news": 0.9,
            "unknown": 0.5
        }
        
        for result in vector_results:
            source = result['metadata'].get('source', 'unknown')
            base_score = result['score']
            
            # Apply source weight
            weighted_score = base_score * source_weights.get(source, 0.5)
            
            # Filter low-quality matches
            if weighted_score < Config.SIMILARITY_THRESHOLD * 0.7:
                continue
            
            ranked.append({
                "source": source,
                "url": result['metadata'].get('url'),
                "title": result['metadata'].get('title'),
                "text": result['metadata'].get('text'),
                "relevance_score": round(weighted_score, 3),
                "published_date": result['metadata'].get('published_date'),
                "evidence_id": result['id']
            })
        
        # Sort by relevance
        ranked.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return ranked[:Config.MAX_SOURCES_PER_CLAIM]
    
    def _extract_keywords(self, text: str) -> str:
        """Extract key terms for search"""
        # Use LLM to extract keywords
        prompt = f"""
        Extract 3-5 key search terms from this claim for finding relevant news articles.
        Focus on specific entities, events, and locations.
        
        Claim: "{text}"
        
        Return only the keywords separated by spaces.
        """
        
        try:
            response = self.llm.predict(prompt)
            return response.strip()
        except:
            # Fallback: use claim text
            return text[:100]
    
    def create_retrieval_task(self, claim: Dict) -> Task:
        """Create CrewAI task for evidence retrieval"""
        return Task(
            description=f"""
            Find authoritative evidence to verify this claim:
            "{claim['claim_text']}"
            
            Steps:
            1. Search the vector database for similar verified content
            2. If needed, crawl trusted sources (PIB, WHO, AP News)
            3. Rank evidence by relevance and source credibility
            4. Return top 5 most relevant pieces of evidence
            
            Prioritize:
            - Government sources (PIB, IMD, NDMA)
            - International organizations (WHO)
            - Reputable news agencies (Reuters, AP)
            """,
            agent=self.agent,
            expected_output="Ranked list of relevant evidence with sources and URLs"
        )
    
    def batch_retrieve(self, claims: List[Dict]) -> List[Dict]:
        """Retrieve evidence for multiple claims"""
        results = []
        
        for claim in claims:
            try:
                evidence_result = self.retrieve_evidence(claim)
                results.append(evidence_result)
            except Exception as e:
                print(f"Error retrieving evidence for claim: {e}")
                results.append({
                    "claim_id": claim.get('id'),
                    "claim_text": claim['claim_text'],
                    "evidence": [],
                    "error": str(e)
                })
        
        return results