"""
Advanced Features Module for TruthGuard
Implements cutting-edge AI capabilities
"""

import asyncio
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import anthropic
from dataclasses import dataclass

# ============================================================================
# ADVANCED FEATURE 1: TEMPORAL CLAIM TRACKING
# ============================================================================

class TemporalTracker:
    """Tracks how claims evolve over time and detects mutations"""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.claim_timeline: Dict[str, List[Dict]] = defaultdict(list)
        self.mutation_patterns: List[Dict] = []
    
    async def track_claim_evolution(self, original_claim: str, new_variant: str) -> Dict:
        """Detect if a claim has mutated to evade detection"""
        
        system_prompt = """You are an expert at detecting semantic similarity and 
        claim mutations. Analyze if two claims are saying the same thing despite 
        different wording."""
        
        prompt = f"""Compare these two claims:

Original: {original_claim}
New Variant: {new_variant}

Are these essentially the same claim? Consider:
1. Core factual assertion
2. Implied meaning
3. Key entities
4. Overall message

Return JSON:
{{
    "is_same_claim": true/false,
    "similarity_score": 0.0-1.0,
    "mutations": ["what changed"],
    "evasion_tactics": ["tactics used to evade detection"]
}}"""

        response = await asyncio.to_thread(
            self.client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        try:
            import json
            return json.loads(result_text.strip())
        except:
            return {"is_same_claim": False, "similarity_score": 0.0}
    
    def add_to_timeline(self, claim_id: str, variant: str, timestamp: datetime):
        """Add claim variant to timeline"""
        self.claim_timeline[claim_id].append({
            "variant": variant,
            "timestamp": timestamp,
            "spread_velocity": self.calculate_spread_velocity(claim_id)
        })
    
    def calculate_spread_velocity(self, claim_id: str) -> float:
        """Calculate how fast a claim is spreading"""
        timeline = self.claim_timeline.get(claim_id, [])
        if len(timeline) < 2:
            return 0.0
        
        # Calculate variants per hour
        time_diff = (timeline[-1]["timestamp"] - timeline[0]["timestamp"]).total_seconds() / 3600
        if time_diff == 0:
            return 0.0
        
        return len(timeline) / time_diff

# ============================================================================
# ADVANCED FEATURE 2: NETWORK ANALYSIS
# ============================================================================

class NetworkAnalyzer:
    """Analyzes spread patterns and identifies coordinated campaigns"""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.spread_graph: Dict[str, List[str]] = defaultdict(list)
        self.bot_indicators: Dict[str, float] = {}
    
    async def detect_coordinated_campaign(self, claim_id: str, sources: List[Dict]) -> Dict:
        """Detect if claim spread shows signs of coordination"""
        
        # Analyze posting patterns
        timestamps = [s.get("timestamp") for s in sources if s.get("timestamp")]
        
        # Calculate temporal clustering
        temporal_clustering = self.calculate_temporal_clustering(timestamps)
        
        # Analyze source similarity
        source_similarity = await self.analyze_source_patterns(sources)
        
        campaign_score = (temporal_clustering + source_similarity) / 2
        
        return {
            "is_coordinated": campaign_score > 0.7,
            "campaign_score": campaign_score,
            "temporal_clustering": temporal_clustering,
            "source_similarity": source_similarity,
            "indicators": self.identify_coordination_indicators(sources)
        }
    
    def calculate_temporal_clustering(self, timestamps: List[datetime]) -> float:
        """Calculate if posts are suspiciously clustered in time"""
        if len(timestamps) < 3:
            return 0.0
        
        timestamps = sorted(timestamps)
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps)-1)]
        
        if not intervals:
            return 0.0
        
        # Low variance = high clustering = suspicious
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 1.0
        
        coefficient_of_variation = std_interval / mean_interval
        
        # Inverse: lower CV = higher clustering score
        clustering_score = max(0, 1 - coefficient_of_variation)
        
        return min(clustering_score, 1.0)
    
    async def analyze_source_patterns(self, sources: List[Dict]) -> float:
        """Analyze if sources show similar patterns (bot behavior)"""
        
        if len(sources) < 3:
            return 0.0
        
        # Check for similar language patterns
        contents = [s.get("content", "") for s in sources]
        
        system_prompt = """Analyze if these texts show signs of being generated by 
        the same source or bots. Look for: identical phrasing, templates, 
        unnatural similarity."""
        
        prompt = f"""Analyze these {len(contents)} posts for bot-like similarity:

{chr(10).join([f"{i+1}. {c[:200]}" for i, c in enumerate(contents[:5])])}

Return similarity score 0.0-1.0 where 1.0 = definitely coordinated/bots."""

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text.strip()
            # Extract number from response
            import re
            match = re.search(r'0?\.\d+|1\.0', text)
            if match:
                return float(match.group())
        except:
            pass
        
        return 0.0
    
    def identify_coordination_indicators(self, sources: List[Dict]) -> List[str]:
        """Identify specific indicators of coordination"""
        indicators = []
        
        # Check for identical domains
        domains = [s.get("domain", "") for s in sources]
        if len(set(domains)) < len(domains) * 0.3:
            indicators.append("High domain repetition")
        
        # Check for posting time clustering
        times = [s.get("timestamp") for s in sources if s.get("timestamp")]
        if times:
            time_range = (max(times) - min(times)).total_seconds()
            if time_range < 3600 and len(times) > 5:  # Many posts in <1 hour
                indicators.append("Rapid coordinated posting")
        
        # Check for identical content
        contents = [s.get("content", "") for s in sources]
        unique_contents = len(set(contents))
        if unique_contents < len(contents) * 0.5:
            indicators.append("High content duplication")
        
        return indicators

# ============================================================================
# ADVANCED FEATURE 3: PREDICTIVE MODELING
# ============================================================================

class PredictiveModeler:
    """Predicts future spread and impact of claims"""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.historical_data: List[Dict] = []
    
    async def predict_spread(self, claim: Dict, current_metrics: Dict) -> Dict:
        """Predict how a claim will spread"""
        
        system_prompt = """You are an expert in viral content dynamics and 
        information spread. Predict how content will spread based on characteristics."""
        
        prompt = f"""Predict the spread of this claim:

Claim characteristics:
- Viral score: {current_metrics.get('viral_score', 0)}
- Current reach: {current_metrics.get('current_reach', 0)} sources
- Emotional valence: {current_metrics.get('sentiment', 0)}
- Controversy level: {current_metrics.get('controversy', 0)}
- Time since detection: {current_metrics.get('age_hours', 0)} hours

Based on these factors, predict:
1. Peak reach (number of sources in 24 hours)
2. Spread velocity (sources per hour)
3. Longevity (days until decline)
4. Demographics most affected

Return as JSON:
{{
    "predicted_peak_reach": 1000,
    "spread_velocity": 50.0,
    "longevity_days": 3,
    "primary_demographics": ["demographic1"],
    "confidence": 0.75
}}"""

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            result_text = response.content[0].text.strip()
            return json.loads(result_text)
        except:
            return {
                "predicted_peak_reach": 0,
                "spread_velocity": 0,
                "longevity_days": 0,
                "confidence": 0
            }
    
    async def predict_impact(self, claim: Dict, verification_result: Dict) -> Dict:
        """Predict real-world impact of misinformation"""
        
        system_prompt = """You are an expert in misinformation impact assessment.
        Predict potential real-world consequences."""
        
        prompt = f"""Assess potential impact:

Claim: {claim.get('text', '')}
Status: {verification_result.get('status', '')}
Confidence: {verification_result.get('confidence', 0)}
Current spread: {claim.get('viral_score', 0)}

Predict:
1. Potential harm types (health, financial, social, political)
2. Scale of impact (individual, community, national)
3. Urgency (hours until critical)
4. Required intervention level

Return as JSON with specific predictions."""

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            result_text = response.content[0].text.strip()
            return json.loads(result_text)
        except:
            return {"harm_types": [], "scale": "unknown", "urgency_hours": 0}

# ============================================================================
# ADVANCED FEATURE 4: MULTI-MODAL ANALYSIS
# ============================================================================

class MultiModalAnalyzer:
    """Analyzes images, videos, and audio for deepfakes and manipulation"""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
    
    async def analyze_image(self, image_data: bytes, claim_text: str) -> Dict:
        """Analyze image for manipulation and consistency with claim"""
        
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        system_prompt = """You are an expert in detecting image manipulation, 
        deepfakes, and misleading visual content. Analyze images for authenticity 
        and context."""
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Analyze this image associated with the claim: "{claim_text}"

Check for:
1. Signs of manipulation (editing, AI generation, deepfake)
2. Context consistency (does image support the claim?)
3. Metadata analysis (if visible)
4. Reverse image search indicators

Return detailed analysis as JSON:
{{
    "manipulation_detected": true/false,
    "manipulation_confidence": 0.0-1.0,
    "manipulation_signs": ["sign1", "sign2"],
    "context_match": true/false,
    "authenticity_score": 0.0-1.0,
    "recommendations": ["recommendation1"]
}}"""
                        }
                    ]
                }]
            )
            
            import json
            result_text = response.content[0].text.strip()
            return json.loads(result_text)
        except Exception as e:
            print(f"Image analysis error: {e}")
            return {
                "manipulation_detected": False,
                "manipulation_confidence": 0.0,
                "authenticity_score": 0.5
            }
    
    async def detect_deepfake_indicators(self, image_data: bytes) -> Dict:
        """Specific deepfake detection"""
        
        # This would integrate with specialized deepfake detection models
        # For now, using Claude's visual understanding
        
        return await self.analyze_image(image_data, "deepfake analysis")

# ============================================================================
# ADVANCED FEATURE 5: CONTEXTUAL EXPLANATION ENGINE
# ============================================================================

class ExplanationEngine:
    """Generates adaptive explanations based on audience"""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
    
    async def generate_explanation(
        self, 
        claim: str, 
        verification: Dict, 
        audience_level: str = "general"
    ) -> str:
        """Generate audience-appropriate explanation"""
        
        audience_prompts = {
            "general": "Explain for average adult readers with no special knowledge",
            "technical": "Explain for technically knowledgeable audience with details",
            "young": "Explain for young readers (age 13-17) in simple, relatable terms",
            "expert": "Explain for subject matter experts with full technical detail"
        }
        
        system_prompt = f"""You are a skilled communicator who adapts explanations 
        to different audiences. {audience_prompts.get(audience_level, audience_prompts['general'])}"""
        
        prompt = f"""Explain this misinformation verdict:

Claim: {claim}
Verdict: {verification.get('status', 'unknown')}
Confidence: {verification.get('confidence', 0)}
Evidence: {verification.get('summary', '')}

Create a clear, accurate explanation that helps the audience understand:
1. What the claim says
2. Why it's true/false/misleading
3. What the evidence shows
4. Why this matters
5. What to do with this information

Keep it concise but complete."""

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except:
            return "Unable to generate explanation at this time."
    
    async def generate_counter_narrative(self, false_claim: str, facts: List[str]) -> str:
        """Generate effective counter-narrative to combat misinformation"""
        
        system_prompt = """You are an expert in effective communication and 
        counter-messaging. Create compelling counter-narratives that don't reinforce 
        the original misinformation."""
        
        prompt = f"""Create an effective counter-narrative for this false claim:

False Claim: {false_claim}

Facts to emphasize:
{chr(10).join([f"- {fact}" for fact in facts])}

Best practices:
- Don't repeat the false claim prominently
- Lead with facts
- Use emotional resonance
- Make it shareable
- Keep it concise

Create the counter-narrative:"""

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except:
            return ""

# ============================================================================
# INTEGRATION CLASS
# ============================================================================

class AdvancedFeatureManager:
    """Manages all advanced features"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Initialize all advanced features
        self.temporal_tracker = TemporalTracker(self.client)
        self.network_analyzer = NetworkAnalyzer(self.client)
        self.predictive_modeler = PredictiveModeler(self.client)
        self.multimodal_analyzer = MultiModalAnalyzer(self.client)
        self.explanation_engine = ExplanationEngine(self.client)
    
    async def full_analysis(self, claim: Dict, sources: List[Dict]) -> Dict:
        """Run all advanced analyses on a claim"""
        
        results = {}
        
        # Run analyses in parallel
        tasks = [
            self.network_analyzer.detect_coordinated_campaign(
                claim.get('id', ''), sources
            ),
            self.predictive_modeler.predict_spread(
                claim, 
                {'viral_score': claim.get('viral_score', 0)}
            ),
            self.explanation_engine.generate_explanation(
                claim.get('text', ''),
                {'status': claim.get('status', ''), 'confidence': claim.get('confidence', 0)},
                'general'
            )
        ]
        
        campaign_analysis, spread_prediction, explanation = await asyncio.gather(*tasks)
        
        return {
            "campaign_analysis": campaign_analysis,
            "spread_prediction": spread_prediction,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def demo_advanced_features():
    """Demonstrate advanced features"""
    
    # Initialize (replace with actual API key)
    manager = AdvancedFeatureManager("your-api-key")
    
    # Example claim
    claim = {
        'id': 'test123',
        'text': 'Breaking: Major policy change announced',
        'viral_score': 0.8,
        'status': 'false',
        'confidence': 0.85
    }
    
    sources = [
        {'domain': 'example.com', 'timestamp': datetime.now(), 'content': 'Test content'},
        {'domain': 'example.com', 'timestamp': datetime.now(), 'content': 'Test content'},
    ]
    
    # Run full analysis
    results = await manager.full_analysis(claim, sources)
    
    print("Advanced Analysis Results:")
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(demo_advanced_features())