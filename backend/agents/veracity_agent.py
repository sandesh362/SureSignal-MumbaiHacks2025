from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from transformers import pipeline
from typing import Dict, List
import numpy as np
from services.llm_service import LLMService

class VeracityAgent:
    def __init__(self):
        llm_service = LLMService()
        self.llm = llm_service.get_chat_llm(model="llama-local", temperature=0.1, system_prompt="You are an expert fact verification assistant.")
        # Load NLI model for entailment checking
        try:
            self.nli_pipeline = pipeline(
                "text-classification",
                model="microsoft/deberta-v3-base-mnli",
                device=-1  # CPU
            )
        except:
            print("Warning: NLI model not loaded, using LLM only")
            self.nli_pipeline = None
        
        self.agent = Agent(
            role="Fact Verification Specialist",
            goal="Determine the veracity of claims using NLP entailment and evidence analysis",
            backstory="""You are an expert fact-checker with deep knowledge of 
            natural language inference. You can determine if evidence supports, 
            contradicts, or is neutral to a claim. You provide confidence scores 
            and clear reasoning.""",
            llm=self.llm,
            verbose=True
        )
    
    def verify_claim(self, claim_text: str, evidence_list: List[Dict]) -> Dict:
        """Verify a claim against evidence"""
        
        if not evidence_list:
            return {
                "verdict": "UNVERIFIED",
                "confidence": 0.0,
                "reasoning": "No evidence found to verify this claim.",
                "evidence_used": []
            }
        
        # Run entailment checks
        entailment_results = []
        
        for evidence in evidence_list:
            if not evidence.get('text'):
                continue
            
            # Check entailment using NLI model
            nli_result = self._check_entailment_nli(claim_text, evidence['text'])
            
            # Check using LLM for nuanced understanding
            llm_result = self._check_entailment_llm(claim_text, evidence)
            
            entailment_results.append({
                "evidence": evidence,
                "nli_label": nli_result['label'],
                "nli_score": nli_result['score'],
                "llm_assessment": llm_result,
                "combined_score": (nli_result['score'] + llm_result['score']) / 2
            })
        
        # Aggregate results
        verdict_data = self._aggregate_verdicts(entailment_results)
        
        return {
            "verdict": verdict_data['verdict'],
            "confidence": verdict_data['confidence'],
            "reasoning": verdict_data['reasoning'],
            "evidence_used": verdict_data['evidence_used'],
            "detailed_checks": entailment_results
        }
    
    def _check_entailment_nli(self, claim: str, evidence: str) -> Dict:
        """Check entailment using NLI model"""
        if not self.nli_pipeline:
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            # Truncate to model max length
            claim = claim[:256]
            evidence = evidence[:512]
            
            result = self.nli_pipeline(f"{claim} [SEP] {evidence}")
            
            label_map = {
                "ENTAILMENT": "SUPPORTS",
                "CONTRADICTION": "CONTRADICTS",
                "NEUTRAL": "NEUTRAL"
            }
            
            return {
                "label": label_map.get(result[0]['label'], "NEUTRAL"),
                "score": result[0]['score']
            }
        except Exception as e:
            print(f"NLI error: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def _check_entailment_llm(self, claim: str, evidence: Dict) -> Dict:
        """Check entailment using LLM for nuanced understanding"""
        prompt = f"""
        Determine if the evidence supports, contradicts, or is neutral to the claim.
        
        Claim: "{claim}"
        
        Evidence from {evidence['source']}:
        Title: {evidence.get('title', 'N/A')}
        Text: {evidence.get('text', '')[:500]}
        
        Analyze:
        1. Does the evidence directly support the claim?
        2. Does the evidence contradict the claim?
        3. Is the evidence neutral or insufficient?
        
        Respond with ONE of: SUPPORTS, CONTRADICTS, NEUTRAL
        Then provide a confidence score from 0.0 to 1.0
        
        Format:
        VERDICT: [SUPPORTS/CONTRADICTS/NEUTRAL]
        CONFIDENCE: [0.0-1.0]
        REASON: [brief explanation]
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Parse response
            lines = response.strip().split('\n')
            verdict = "NEUTRAL"
            confidence = 0.5
            reason = ""
            
            for line in lines:
                if line.startswith("VERDICT:"):
                    verdict = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
            
            return {
                "label": verdict,
                "score": confidence,
                "reasoning": reason
            }
        except Exception as e:
            print(f"LLM entailment error: {e}")
            return {"label": "NEUTRAL", "score": 0.5, "reasoning": "Error in analysis"}
    
    def _aggregate_verdicts(self, results: List[Dict]) -> Dict:
        """Aggregate multiple entailment results into final verdict"""
        if not results:
            return {
                "verdict": "UNVERIFIED",
                "confidence": 0.0,
                "reasoning": "No evidence available",
                "evidence_used": []
            }
        
        # Count verdicts
        supports = []
        contradicts = []
        neutral = []
        
        for r in results:
            score = r['combined_score']
            label = r['llm_assessment']['label']
            
            if label == "SUPPORTS":
                supports.append((score, r))
            elif label == "CONTRADICTS":
                contradicts.append((score, r))
            else:
                neutral.append((score, r))
        
        # Decision logic
        support_strength = sum(s[0] for s in supports) / len(results)
        contradict_strength = sum(c[0] for c in contradicts) / len(results)
        
        # Determine verdict
        if support_strength > 0.6 and support_strength > contradict_strength * 2:
            verdict = "TRUE"
            confidence = support_strength
            top_evidence = sorted(supports, key=lambda x: x[0], reverse=True)[:3]
            reasoning = f"Multiple sources support this claim. {top_evidence[0][1]['llm_assessment']['reasoning']}"
        
        elif contradict_strength > 0.6 and contradict_strength > support_strength * 2:
            verdict = "FALSE"
            confidence = contradict_strength
            top_evidence = sorted(contradicts, key=lambda x: x[0], reverse=True)[:3]
            reasoning = f"Evidence contradicts this claim. {top_evidence[0][1]['llm_assessment']['reasoning']}"
        
        elif support_strength > 0.4 and contradict_strength > 0.4:
            verdict = "MISLEADING"
            confidence = max(support_strength, contradict_strength)
            top_evidence = sorted(results, key=lambda x: x['combined_score'], reverse=True)[:3]
            reasoning = "Mixed evidence - some sources support while others contradict. The claim may be partially true or taken out of context."
        
        else:
            verdict = "UNVERIFIED"
            confidence = max(support_strength, contradict_strength)
            top_evidence = results[:3]
            reasoning = "Insufficient strong evidence to verify this claim conclusively."
        
        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "evidence_used": [e[1]['evidence'] for e in top_evidence]
        }
    
    def create_verification_task(self, claim: Dict, evidence: List[Dict]) -> Task:
        """Create CrewAI task for verification"""
        return Task(
            description=f"""
            Verify this claim using natural language inference:
            
            Claim: "{claim['claim_text']}"
            
            Evidence: {len(evidence)} sources available
            
            For each piece of evidence:
            1. Determine if it SUPPORTS, CONTRADICTS, or is NEUTRAL
            2. Calculate confidence score
            3. Provide reasoning
            
            Then aggregate all results into a final verdict:
            - TRUE: Strong evidence supports the claim
            - FALSE: Strong evidence contradicts the claim
            - MISLEADING: Mixed or partial evidence
            - UNVERIFIED: Insufficient evidence
            
            Include confidence score (0-1) and clear explanation.
            """,
            agent=self.agent,
            expected_output="Final verdict with confidence score, reasoning, and supporting evidence"
        )
    
    def batch_verify(self, claims_with_evidence: List[Dict]) -> List[Dict]:
        """Verify multiple claims"""
        results = []
        
        for item in claims_with_evidence:
            claim_text = item['claim_text']
            evidence = item.get('evidence', [])
            
            verification = self.verify_claim(claim_text, evidence)
            verification['claim_text'] = claim_text
            verification['claim_id'] = item.get('claim_id')
            
            results.append(verification)
        
        return results