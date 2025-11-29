from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List
from services.llm_service import LLMService

class ExplanationAgent:
    def __init__(self):
        llm_service = LLMService()
        # Use local llama for detailed explanations (or configured default)
        self.llm_detailed = llm_service.get_chat_llm(model="llama-local", temperature=0.4,
                                                    system_prompt="You are a professional detailed explainer.")
        # Use local llama smaller / faster for simple ELI12 (or same server but different prompt/model)
        self.llm_simple = llm_service.get_chat_llm(model="llama-local", temperature=0.6, system_prompt="Explain simply like to a 12-year-old.")
        
        self.agent = Agent(
            role="Explanation and Communication Specialist",
            goal="Convert technical verification results into clear, accessible explanations",
            backstory="""You are an expert communicator who can explain complex 
            fact-checking results to any audience. You excel at creating both 
            detailed explanations and simple 'Explain Like I'm 12' versions. 
            You always cite sources properly.""",
            llm=self.llm_detailed,
            verbose=True
        )
    
    def generate_explanation(self, verification: Dict) -> Dict:
        """Generate comprehensive and ELI12 explanations"""
        
        # Generate detailed explanation
        detailed = self._generate_detailed_explanation(verification)
        
        # Generate ELI12 version
        eli12 = self._generate_eli12_explanation(verification)
        
        # Format citations
        citations = self._format_citations(verification.get('evidence_used', []))
        
        # Generate verdict summary
        summary = self._generate_summary(verification)
        
        return {
            "summary": summary,
            "detailed_explanation": detailed,
            "eli12_explanation": eli12,
            "citations": citations,
            "verdict": verification['verdict'],
            "confidence": verification['confidence']
        }
    
    def _generate_detailed_explanation(self, verification: Dict) -> str:
        """Generate detailed explanation for informed readers"""
        verdict = verification['verdict']
        confidence = verification['confidence']
        reasoning = verification.get('reasoning', '')
        evidence = verification.get('evidence_used', [])
        
        prompt = f"""
        Create a detailed, professional explanation of this fact-check result.
        
        Verdict: {verdict}
        Confidence: {confidence * 100:.0f}%
        Reasoning: {reasoning}
        
        Evidence sources: {len(evidence)}
        {self._format_evidence_summary(evidence)}
        
        Write a 3-4 sentence explanation that:
        1. States the verdict clearly
        2. Explains the reasoning
        3. References key evidence
        4. Notes any caveats or limitations
        
        Be authoritative but fair. Use professional language.
        """
        
        try:
            explanation = self.llm_detailed.predict(prompt)
            return explanation.strip()
        except Exception as e:
            print(f"Error generating detailed explanation: {e}")
            return f"{verdict}: {reasoning}"
    
    def _generate_eli12_explanation(self, verification: Dict) -> str:
        """Generate simple explanation for general audience"""
        verdict = verification['verdict']
        confidence = verification['confidence']
        reasoning = verification.get('reasoning', '')
        
        prompt = f"""
        Explain this fact-check result to a 12-year-old.
        
        Verdict: {verdict}
        Confidence: {confidence * 100:.0f}%
        Reasoning: {reasoning}
        
        Use simple words and short sentences. No jargon.
        Make it engaging and easy to understand.
        2-3 sentences maximum.
        """
        
        try:
            explanation = self.llm_simple.predict(prompt)
            return explanation.strip()
        except Exception as e:
            print(f"Error generating ELI12: {e}")
            return self._fallback_eli12(verdict, reasoning)
    
    def _fallback_eli12(self, verdict: str, reasoning: str) -> str:
        """Simple fallback ELI12 explanation"""
        templates = {
            "TRUE": f"This claim is true. {reasoning[:100]}",
            "FALSE": f"This claim is false. {reasoning[:100]}",
            "MISLEADING": f"This claim is misleading. {reasoning[:100]}",
            "UNVERIFIED": f"We couldn't verify this claim. {reasoning[:100]}"
        }
        return templates.get(verdict, reasoning[:150])
    
    def _generate_summary(self, verification: Dict) -> str:
        """Generate one-line summary"""
        verdict = verification['verdict']
        confidence = verification['confidence']
        
        templates = {
            "TRUE": f"✓ VERIFIED AS TRUE ({confidence*100:.0f}% confidence)",
            "FALSE": f"✗ VERIFIED AS FALSE ({confidence*100:.0f}% confidence)",
            "MISLEADING": f"⚠ MISLEADING ({confidence*100:.0f}% confidence)",
            "UNVERIFIED": f"? UNVERIFIED ({confidence*100:.0f}% confidence)"
        }
        
        return templates.get(verdict, f"{verdict} ({confidence*100:.0f}%)")
    
    def _format_evidence_summary(self, evidence: List[Dict]) -> str:
        """Create summary of evidence sources"""
        if not evidence:
            return "No evidence found"
        
        summaries = []
        for e in evidence[:3]:
            source = e.get('source', 'Unknown')
            title = e.get('title', 'Untitled')
            summaries.append(f"- {source.upper()}: {title[:60]}")
        
        return "\n".join(summaries)
    
    def _format_citations(self, evidence: List[Dict]) -> List[Dict]:
        """Format evidence into proper citations"""
        citations = []
        
        for i, e in enumerate(evidence, 1):
            citation = {
                "number": i,
                "source": e.get('source', 'Unknown').upper(),
                "title": e.get('title', 'Untitled'),
                "url": e.get('url', '#'),
                "published_date": e.get('published_date'),
                "relevance": e.get('relevance_score', 0)
            }
            citations.append(citation)
        
        return citations
    
    def format_bot_reply(self, claim_text: str, explanation: Dict) -> str:
        """Format explanation as bot reply (Twitter/Reddit)"""
        verdict = explanation['verdict']
        confidence = explanation['confidence']
        eli12 = explanation['eli12_explanation']
        citations = explanation['citations']
        
        # Verdict emoji
        emoji_map = {
            "TRUE": "✅",
            "FALSE": "❌",
            "MISLEADING": "⚠️",
            "UNVERIFIED": "❓"
        }
        emoji = emoji_map.get(verdict, "ℹ️")
        
        # Build reply
        reply = f"{emoji} Verdict: {verdict} ({confidence*100:.0f}% confidence)\n\n"
        reply += f"📝 {eli12}\n\n"
        
        if citations:
            reply += "📚 Sources:\n"
            for cite in citations[:3]:
                reply += f"• {cite['source']}: {cite['title'][:50]}...\n"
                reply += f"  {cite['url']}\n"
        
        reply += "\n🔗 Full report: [link to web portal]"
        
        return reply
    
    def format_web_response(self, claim_text: str, explanation: Dict) -> Dict:
        """Format explanation for web portal"""
        return {
            "claim": claim_text,
            "verdict": explanation['verdict'],
            "confidence": explanation['confidence'],
            "summary": explanation['summary'],
            "detailed_explanation": explanation['detailed_explanation'],
            "eli12_explanation": explanation['eli12_explanation'],
            "citations": explanation['citations'],
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def create_explanation_task(self, verification: Dict) -> Task:
        """Create CrewAI task for explanation generation"""
        return Task(
            description=f"""
            Generate clear explanations for this verification result:
            
            Verdict: {verification['verdict']}
            Confidence: {verification['confidence']}
            Reasoning: {verification.get('reasoning')}
            
            Create:
            1. Professional detailed explanation (3-4 sentences)
            2. Simple ELI12 version (2-3 sentences)
            3. Formatted citations for all evidence
            4. One-line summary with confidence score
            
            The explanations should be clear, fair, and cite sources properly.
            """,
            agent=self.agent,
            expected_output="Complete explanation package with multiple formats"
        )
    
    def batch_explain(self, verifications: List[Dict]) -> List[Dict]:
        """Generate explanations for multiple verifications"""
        results = []
        
        for verification in verifications:
            try:
                explanation = self.generate_explanation(verification)
                explanation['claim_text'] = verification.get('claim_text')
                explanation['claim_id'] = verification.get('claim_id')
                results.append(explanation)
            except Exception as e:
                print(f"Error explaining verification: {e}")
                results.append({
                    "claim_text": verification.get('claim_text'),
                    "error": str(e),
                    "verdict": verification.get('verdict'),
                    "confidence": verification.get('confidence')
                })
        
        return results