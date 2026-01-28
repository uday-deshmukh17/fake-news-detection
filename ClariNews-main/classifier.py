import os
import json
import logging
import requests
import subprocess
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter
import numpy as np
from config import app_config, Config
from source_search import search_and_extract, extract_entities, calculate_source_credibility
from translator import translate_text, detect_language

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EnhancedFactChecker:
    """Production-ready fact checker with ML and advanced verification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_name = "llama3.2"
        self.ollama_url = "http://localhost:11434/api/generate"
        
        self.gpu_layers = self._calculate_optimal_gpu_layers()
        self.use_gpu = self.gpu_layers > 0
        
        # Enhanced trusted sources with credibility scores
        self.trusted_sources = {
            # Tier 1: Highest credibility (0.95-1.0)
            'reuters.com': 1.0,
            'apnews.com': 1.0,
            'bbc.com': 0.98,
            'bbc.co.uk': 0.98,
            'theguardian.com': 0.95,
            'nytimes.com': 0.95,
            'washingtonpost.com': 0.95,
            'aljazeera.com': 0.95,
            'thehindu.com': 0.95,
            'indianexpress.com': 0.95,
            
            # Tier 2: High credibility (0.85-0.94)
            'cnn.com': 0.92,
            'timesofindia.indiatimes.com': 0.90,
            'hindustantimes.com': 0.90,
            'ndtv.com': 0.90,
            'news18.com': 0.88,
            'indiatoday.in': 0.88,
            'bloomberg.com': 0.92,
            'forbes.com': 0.90,
            'economist.com': 0.92,
            'cnbc.com': 0.88,
            'ft.com': 0.92,
            'wsj.com': 0.92,
            
            # Tier 3: Reliable (0.75-0.84)
            'moneycontrol.com': 0.85,
            'economictimes.indiatimes.com': 0.85,
            'espn.com': 0.82,
            'espncricinfo.com': 0.82,
            'cricbuzz.com': 0.80,
            'wikipedia.org': 0.78,
            
            # Tier 4: Fact-checkers (0.9-0.95)
            'factcheck.org': 0.95,
            'snopes.com': 0.95,
            'politifact.com': 0.95,
            'boomlive.in': 0.90,
            'altnews.in': 0.90,
        }
        
        self._check_ollama()

    def _calculate_optimal_gpu_layers(self) -> int:
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.total,memory.free', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    total_mem, free_mem = map(int, lines[0].split(', '))
                    logger.info(f"GPU: {total_mem}MB total, {free_mem}MB free")
                    
                    if total_mem >= 16000: return 32
                    elif total_mem >= 12000: return 24
                    elif total_mem >= 8000: return 16
                    else: return 8
        except:
            logger.warning("No NVIDIA GPU detected")
        return 8

    def _check_ollama(self):
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError("Ollama not responding")
            models = resp.json().get("models", [])
            names = [m.get("name", "") for m in models]
            if not any(self.model_name in n for n in names):
                raise ValueError(f"Model {self.model_name} not found. Run: ollama pull {self.model_name}")
            
            logger.info(f"Model: {self.model_name}, GPU layers: {self.gpu_layers}")
        except requests.exceptions.ConnectionError:
            logger.error("Ollama not running. Start: ollama serve")
            raise

        # Add this method to your EnhancedFactChecker class in classifier.py

    def classify(self, news_text: str, category: str = None, language: str = "en") -> Dict[str, Any]:
        """Main classification method with enhanced verification"""
        ts = datetime.now().isoformat()
        original_language = language.lower()
        original_text = news_text
        
        if not category or category.lower() == 'all':
            category = 'all'
        
        # Translation
        if original_language != "en":
            detected_lang = detect_language(news_text)
            logger.info(f"Detected: {detected_lang}, Requested: {original_language}")
            news_text = translate_text(news_text, source='auto', target='en')
            logger.info(f"Translated: {news_text[:100]}...")
        
        # Extract entities and claims BEFORE searching
        entities = extract_entities(news_text)
        logger.info(f"Extracted entities: {entities}")
        
        # Enhanced web search with better scraping
        logger.info(f"Starting enhanced search for: {news_text[:50]}...")
        search_results = search_and_extract(news_text, category, max_results=15)
        
        if len(search_results) < 2:
            logger.warning(f"Only {len(search_results)} sources found")
            result = {
                "search_error": f"Insufficient sources ({len(search_results)}/2). Cannot verify.",
                "label": 'UNVERIFIED',
                "confidence": 0.5,
                "sources_found": len(search_results),
                "timestamp": ts,
                "explanation": "Not enough credible sources to verify this claim.",
                "web_evidence": []  # Add empty web_evidence
            }
            if original_language != "en":
                result = self._translate_response(result, original_language)
            return result
        
        logger.info(f"Found {len(search_results)} sources")
        
        # Multi-layer verification
        verification_result = self._multi_layer_verification(news_text, search_results, entities, category)
        
        # Build LLM prompt with all evidence
        prompt = self._build_comprehensive_prompt(news_text, search_results, verification_result, category)
        
        # Call Ollama
        try:
            logger.info(f"Analyzing with {self.model_name}...")
            options = self._get_inference_options()
            resp = self._post_ollama(prompt, options)
            
            if resp.status_code != 200 and self.use_gpu:
                logger.warning("GPU failed, using CPU")
                self.use_gpu = False
                self.gpu_layers = 0
                options = self._get_cpu_only_options()
                resp = self._post_ollama(prompt, options)
            
            if resp.status_code != 200:
                raise Exception(f"Ollama failed: {resp.text[:200]}")
            
            result_data = resp.json()
            generated = result_data.get("response", "")
            
            if "eval_count" in result_data:
                tok = result_data.get('eval_count', 0)
                sec = result_data.get('eval_duration', 0) / 1e9
                if sec:
                    logger.info(f"Generated {tok} tokens in {sec:.2f}s ({tok/sec:.1f} tok/s)")
        
        except Exception as e:
            logger.error(f"Model error: {e}")
            result = self._error_response(str(e)[:100], ts)
            if original_language != "en":
                result = self._translate_response(result, original_language)
            return result
        
        # Parse and validate LLM output
        parsed = self._parse_and_validate(generated, verification_result)
        
        # ⭐ BUILD ENHANCED WEB EVIDENCE STRUCTURE ⭐
        web_evidence = []
        for idx, source in enumerate(search_results, 1):
            web_evidence.append({
                "source_number": idx,
                "title": source.get('title', 'No Title'),
                "snippet": source.get('snippet', '')[:300],  # First 300 chars of snippet
                "full_content": source.get('full_content', '')[:5000],  # First 5000 chars
                "url": source.get('url', ''),
                "domain": source.get('domain', ''),
                "trusted": source.get('credibility', 0) >= 0.8,
                "credibility_score": source.get('credibility', 0),
                "content_length": len(source.get('full_content', ''))
            })
        
        # ⭐ BUILD ENHANCED SOURCE ANALYSIS ⭐
        source_analysis = {
            "contradictions": [],
            "confirmations": [],
            "supporting_sources": verification_result['supporting_sources'],
            "contradicting_sources": verification_result['contradicting_sources'],
            "neutral_sources": verification_result['neutral_sources'],
            "fake_indicators": [],
            "claims_extracted": {
                "numerical_claims": entities.get('numbers', []),
                "entities": list(set(entities.get('persons', []) + entities.get('organizations', []))),
                "temporal_claims": entities.get('dates', [])
            },
            "consensus_ratio": verification_result['consensus_score']
        }
        
        # Add contradictions from verification
        for contradiction in verification_result.get('contradictions', []):
            source_analysis['contradictions'].append({
                "source": contradiction.get('source_idx', 0) + 1,
                "type": contradiction.get('type', 'unknown'),
                "claimed": contradiction.get('claimed', ''),
                "found": contradiction.get('found', ''),
                "evidence": search_results[contradiction.get('source_idx', 0)].get('full_content', '')[:200]
            })
        
        # Add confirmations from semantic analysis
        for idx, alignment in enumerate(verification_result.get('semantic_analysis', {}).get('source_alignments', [])):
            if alignment.get('has_confirmation'):
                source_analysis['confirmations'].append({
                    "source": idx + 1,
                    "type": "explicit_confirmation",
                    "evidence": search_results[idx].get('full_content', '')[:200]
                })
        
        # Build final response with ALL data
        parsed.update({
            "sources": [s.get('url') for s in search_results if s.get('url')],
            "category": category or "all",
            "timestamp": ts,
            "total_sources_checked": len(search_results),
            "trusted_sources": sum(1 for s in search_results if s.get('credibility', 0) >= 0.8),
            "verification_layers": verification_result,
            "model_used": self.model_name,
            "language": original_language,
            "entities_extracted": entities,
            "web_evidence": web_evidence,  # ⭐ ADD FULL WEB EVIDENCE
            "source_analysis": source_analysis,  # ⭐ ADD SOURCE ANALYSIS
            "verification_quality": "HIGH" if verification_result['consensus_score'] >= 0.7 else 
                                "MEDIUM" if verification_result['consensus_score'] >= 0.5 else "LOW"
        })
        
        if original_language != "en":
            parsed = self._translate_response(parsed, original_language)
        
        logger.info(f"FINAL: {parsed['label']} (conf: {parsed['confidence']:.2f})")
        return parsed

    def _multi_layer_verification(self, claim: str, sources: List[dict], entities: dict, category: str) -> Dict[str, Any]:
        """Multi-layer verification using multiple algorithms"""
        
        # Layer 1: Source credibility scoring
        credibility_scores = []
        for source in sources:
            url = source.get('url', '')
            domain = self._extract_domain(url)
            base_score = self.trusted_sources.get(domain, 0.5)
            
            # Additional credibility checks
            content_length = len(source.get('full_content', ''))
            if content_length > 1000:
                base_score += 0.05
            if content_length > 2000:
                base_score += 0.05
            
            credibility_scores.append(min(1.0, base_score))
        
        avg_credibility = np.mean(credibility_scores) if credibility_scores else 0.5
        
        # Layer 2: Entity matching
        entity_matches = self._check_entity_matches(claim, sources, entities)
        
        # Layer 3: Numerical verification
        numerical_verification = self._verify_numerical_claims(claim, sources)
        
        # Layer 4: Semantic similarity
        semantic_analysis = self._semantic_verification(claim, sources)
        
        # Layer 5: Contradiction detection
        contradictions = self._detect_contradictions(claim, sources, entities)
        
        # Layer 6: Temporal verification
        temporal_check = self._verify_temporal_claims(claim, sources)
        
        # Calculate consensus
        supporting = sum(1 for s in semantic_analysis['source_alignments'] if s['alignment'] > 0.6)
        contradicting = len(contradictions)
        neutral = len(sources) - supporting - contradicting
        
        consensus_score = supporting / len(sources) if sources else 0
        
        return {
            "avg_source_credibility": avg_credibility,
            "entity_match_score": entity_matches['match_score'],
            "entity_details": entity_matches['details'],
            "numerical_verification": numerical_verification,
            "semantic_analysis": semantic_analysis,
            "contradictions": contradictions,
            "temporal_verification": temporal_check,
            "supporting_sources": supporting,
            "contradicting_sources": contradicting,
            "neutral_sources": neutral,
            "consensus_score": consensus_score,
            "credibility_scores": credibility_scores
        }

    def _check_entity_matches(self, claim: str, sources: List[dict], entities: dict) -> Dict[str, Any]:
        """Check if key entities appear in sources"""
        claim_lower = claim.lower()
        
        # Extract key entities from claim
        key_entities = []
        if entities.get('persons'):
            key_entities.extend(entities['persons'])
        if entities.get('organizations'):
            key_entities.extend(entities['organizations'])
        if entities.get('locations'):
            key_entities.extend(entities['locations'])
        
        if not key_entities:
            # Fallback: extract capitalized words
            key_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
        
        # Check each source
        entity_presence = []
        for source in sources:
            content = source.get('full_content', '').lower()
            title = source.get('title', '').lower()
            full_text = f"{title} {content}"
            
            matches = sum(1 for entity in key_entities if entity.lower() in full_text)
            match_ratio = matches / len(key_entities) if key_entities else 0
            entity_presence.append(match_ratio)
        
        avg_match = np.mean(entity_presence) if entity_presence else 0
        
        return {
            "match_score": avg_match,
            "details": {
                "key_entities": key_entities[:5],
                "source_match_ratios": entity_presence
            }
        }

    def _verify_numerical_claims(self, claim: str, sources: List[dict]) -> Dict[str, Any]:
        """Verify numerical claims across sources"""
        # Extract numbers from claim
        claim_numbers = re.findall(r'\$?\d+(?:\.\d+)?\s*(?:trillion|billion|million|thousand|crore|lakh)?', claim.lower())
        
        if not claim_numbers:
            return {"has_numerical_claims": False}
        
        # Check sources for matching/contradicting numbers
        matches = []
        contradictions = []
        
        for claim_num in claim_numbers:
            found_in_sources = 0
            contradicting_sources = []
            
            for idx, source in enumerate(sources):
                content = source.get('full_content', '').lower()
                
                if claim_num in content:
                    found_in_sources += 1
                else:
                    # Check for contradicting numbers
                    pattern = r'\$?\d+(?:\.\d+)?\s*(?:trillion|billion|million|thousand|crore|lakh)?'
                    source_numbers = re.findall(pattern, content)
                    if source_numbers:
                        contradicting_sources.append({
                            'source_idx': idx,
                            'found_numbers': source_numbers[:3]
                        })
            
            matches.append({
                'claimed': claim_num,
                'confirmed_by': found_in_sources,
                'total_sources': len(sources)
            })
            
            if contradicting_sources:
                contradictions.append({
                    'claimed': claim_num,
                    'contradicting_sources': contradicting_sources[:2]
                })
        
        verification_score = sum(m['confirmed_by'] for m in matches) / (len(matches) * len(sources)) if matches else 0
        
        return {
            "has_numerical_claims": True,
            "claims": claim_numbers,
            "matches": matches,
            "contradictions": contradictions,
            "verification_score": verification_score
        }

    def _semantic_verification(self, claim: str, sources: List[dict]) -> Dict[str, Any]:
        """Semantic analysis of claim against sources"""
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        
        # Remove stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'of', 'for'}
        claim_words = claim_words - stop_words
        
        source_alignments = []
        for idx, source in enumerate(sources):
            content = source.get('full_content', '').lower()
            title = source.get('title', '').lower()
            source_text = f"{title} {content}"
            
            source_words = set(re.findall(r'\b\w+\b', source_text)) - stop_words
            
            # Calculate word overlap
            overlap = len(claim_words & source_words)
            alignment_score = overlap / len(claim_words) if claim_words else 0
            
            # Check for confirmation phrases
            confirmation_phrases = ['confirmed', 'verified', 'announced', 'official', 'according to']
            has_confirmation = any(phrase in source_text for phrase in confirmation_phrases)
            
            # Check for denial phrases
            denial_phrases = ['false', 'fake', 'hoax', 'denied', 'debunked', 'not true']
            has_denial = any(phrase in source_text for phrase in denial_phrases)
            
            source_alignments.append({
                'source_idx': idx,
                'alignment': alignment_score,
                'has_confirmation': has_confirmation,
                'has_denial': has_denial
            })
        
        avg_alignment = np.mean([s['alignment'] for s in source_alignments]) if source_alignments else 0
        
        return {
            "avg_alignment": avg_alignment,
            "source_alignments": source_alignments
        }

    def _detect_contradictions(self, claim: str, sources: List[dict], entities: dict) -> List[Dict[str, Any]]:
        """Detect explicit contradictions"""
        contradictions = []
        claim_lower = claim.lower()
        
        # Check for "X is Y" patterns
        is_pattern = re.search(r'(\w+(?:\s+\w+)*)\s+is\s+(?:the\s+)?(\w+(?:\s+\w+)*)\s+(?:of|in)\s+(\w+(?:\s+\w+)*)', claim_lower)
        
        if is_pattern:
            subject = is_pattern.group(1).strip()
            relationship = is_pattern.group(2).strip()
            obj = is_pattern.group(3).strip()
            
            for idx, source in enumerate(sources):
                content = source.get('full_content', '').lower()
                
                # Look for different relationships
                alt_pattern = rf'{re.escape(subject)}.*?{re.escape(relationship)}.*?(\w+(?:\s+\w+)*)'
                matches = re.findall(alt_pattern, content)
                
                for match in matches:
                    if match != obj and len(match) > 3:
                        contradictions.append({
                            'type': 'relationship_mismatch',
                            'source_idx': idx,
                            'claimed': f"{subject} is {relationship} of {obj}",
                            'found': f"{subject} is {relationship} of {match}"
                        })
                        break
        
        # Check for death claims
        death_keywords = ['died', 'death', 'passed away', 'deceased', 'dead']
        alive_keywords = ['alive', 'living', 'survived']
        
        has_death_claim = any(kw in claim_lower for kw in death_keywords)
        has_alive_claim = any(kw in claim_lower for kw in alive_keywords)
        
        if has_death_claim or has_alive_claim:
            for idx, source in enumerate(sources):
                content = source.get('full_content', '').lower()
                
                if has_alive_claim and any(kw in content for kw in death_keywords):
                    contradictions.append({
                        'type': 'life_status_contradiction',
                        'source_idx': idx,
                        'claimed': 'alive',
                        'found': 'died'
                    })
                elif has_death_claim and any(kw in content for kw in alive_keywords):
                    # Check context - "alive in hearts" is not alive
                    if 'alive in' in content and any(w in content for w in ['hearts', 'memory', 'minds']):
                        continue  # This is figurative, not contradiction
                    contradictions.append({
                        'type': 'life_status_contradiction',
                        'source_idx': idx,
                        'claimed': 'died',
                        'found': 'alive'
                    })
        
        return contradictions

    def _verify_temporal_claims(self, claim: str, sources: List[dict]) -> Dict[str, Any]:
        """Verify temporal claims"""
        # Extract years from claim
        claim_years = re.findall(r'\b(19|20)\d{2}\b', claim)
        
        if not claim_years:
            return {"has_temporal_claims": False}
        
        # Check sources
        confirmations = 0
        for source in sources:
            content = source.get('full_content', '')
            for year in claim_years:
                if year in content:
                    confirmations += 1
                    break
        
        verification_score = confirmations / len(sources) if sources else 0
        
        return {
            "has_temporal_claims": True,
            "claimed_years": claim_years,
            "confirmation_rate": verification_score
        }

    def _build_comprehensive_prompt(self, claim: str, sources: List[dict], verification: Dict, category: str) -> str:
        """Build comprehensive prompt with all verification data"""
        
        system = """You are an EXPERT fact-checker AI. Analyze the claim against multiple sources and verification layers.

VERIFICATION FRAMEWORK:
1. Source Credibility: Higher credibility sources (0.9+) are more reliable
2. Entity Matching: Key entities should appear in multiple sources
3. Numerical Claims: Numbers must match exactly across sources
4. Semantic Alignment: Content should align with claim semantically
5. Contradictions: Any explicit contradictions indicate FAKE
6. Consensus: 70%+ sources supporting = REAL, <50% = FAKE

DECISION RULES:
- If contradictions found OR numerical mismatch → FAKE (0.85+ confidence)
- If consensus_score ≥ 0.7 AND avg_credibility ≥ 0.8 → REAL (0.8+ confidence)
- If consensus_score < 0.5 → FAKE (0.7+ confidence)
- If avg_credibility < 0.6 → UNVERIFIED (0.5-0.6 confidence)

OUTPUT JSON ONLY:
{
  "label": "REAL" or "FAKE" or "UNVERIFIED",
  "confidence": 0.0-1.0,
  "explanation": "Clear explanation with evidence"
}"""

        user_msg = f"CLAIM: {claim}\n\n"
        
        if category:
            user_msg += f"CATEGORY: {category}\n\n"
        
        # Add verification summary
        user_msg += f"VERIFICATION SUMMARY:\n"
        user_msg += f"- Source Credibility: {verification['avg_source_credibility']:.2f}\n"
        user_msg += f"- Entity Match Score: {verification['entity_match_score']:.2f}\n"
        user_msg += f"- Consensus Score: {verification['consensus_score']:.2f}\n"
        user_msg += f"- Supporting: {verification['supporting_sources']}/{len(sources)}\n"
        user_msg += f"- Contradicting: {verification['contradicting_sources']}\n\n"
        
        # Show contradictions
        if verification['contradictions']:
            user_msg += "⚠️ CONTRADICTIONS DETECTED:\n"
            for c in verification['contradictions'][:3]:
                user_msg += f"  - {c['type']}: Claimed '{c['claimed']}', Found '{c['found']}'\n"
            user_msg += "\n"
        
        # Show numerical verification
        if verification['numerical_verification'].get('has_numerical_claims'):
            num_ver = verification['numerical_verification']
            user_msg += f"NUMERICAL VERIFICATION (score: {num_ver['verification_score']:.2f}):\n"
            for m in num_ver['matches'][:3]:
                user_msg += f"  - '{m['claimed']}': confirmed by {m['confirmed_by']}/{m['total_sources']} sources\n"
            user_msg += "\n"
        
        # Show sources
        user_msg += "SOURCES:\n"
        for idx, source in enumerate(sources[:10], 1):
            cred_score = verification['credibility_scores'][idx-1] if idx-1 < len(verification['credibility_scores']) else 0.5
            title = source.get('title', 'No title')
            content = source.get('full_content', '')[:400]
            user_msg += f"\n[Source {idx}] (Credibility: {cred_score:.2f})\n"
            user_msg += f"Title: {title}\n"
            user_msg += f"Content: {content}...\n"
        
        user_msg += "\n\nBased on all evidence, is this claim REAL, FAKE, or UNVERIFIED?"
        
        prompt = f"<s>[INST] {system}\n\n{user_msg} [/INST]"
        return prompt

    def _parse_and_validate(self, raw: str, verification: Dict) -> Dict[str, Any]:
        """Parse LLM output and validate against verification layers"""
        
        # Parse JSON
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return self._fallback_decision(verification)
        
        try:
            parsed = json.loads(m.group(0))
        except:
            return self._fallback_decision(verification)
        
        label = str(parsed.get("label", "UNVERIFIED")).upper()
        confidence = float(parsed.get("confidence", 0.5))
        explanation = str(parsed.get("explanation", ""))
        
        # Validate against verification layers
        
        # Override 1: Strong contradictions
        if verification['contradicting_sources'] > 0:
            label = "FAKE"
            confidence = max(confidence, 0.85)
        
        # Override 2: Low credibility sources
        if verification['avg_source_credibility'] < 0.6:
            label = "UNVERIFIED"
            confidence = min(confidence, 0.6)
        
        # Override 3: Strong consensus with high credibility
        if verification['consensus_score'] >= 0.7 and verification['avg_source_credibility'] >= 0.8:
            if label == "FAKE":
                # LLM says FAKE but sources say REAL - trust sources
                label = "REAL"
                confidence = 0.8
        
        # Override 4: Low consensus
        if verification['consensus_score'] < 0.4:
            label = "FAKE"
            confidence = max(confidence, 0.7)
        
        # Override 5: Numerical contradictions
        if verification['numerical_verification'].get('contradictions'):
            label = "FAKE"
            confidence = max(confidence, 0.9)
        
        # Ensure valid label
        if label not in ["REAL", "FAKE", "UNVERIFIED"]:
            label = "UNVERIFIED"
        
        # Clamp confidence
        confidence = max(0.1, min(1.0, confidence))
        
        return {
            "label": label,
            "confidence": confidence,
            "explanation": explanation
        }

    def _fallback_decision(self, verification: Dict) -> Dict[str, Any]:
        """Make decision based on verification layers when LLM fails"""
        
        # Decision tree based on verification
        if verification['contradicting_sources'] > 0:
            return {
                "label": "FAKE",
                "confidence": 0.85,
                "explanation": f"Found {verification['contradicting_sources']} contradicting sources."
            }
        
        if verification['consensus_score'] >= 0.7 and verification['avg_source_credibility'] >= 0.8:
            return {
                "label": "REAL",
                "confidence": 0.80,
                "explanation": f"{verification['supporting_sources']} credible sources confirm this."
            }
        
        if verification['consensus_score'] < 0.4:
            return {
                "label": "FAKE",
                "confidence": 0.70,
                "explanation": "Low consensus among sources."
            }
        
        if verification['avg_source_credibility'] < 0.6:
            return {
                "label": "UNVERIFIED",
                "confidence": 0.55,
                "explanation": "Sources lack sufficient credibility."
            }
        
        return {
            "label": "UNVERIFIED",
            "confidence": 0.50,
            "explanation": "Insufficient evidence to make determination."
        }

    def _extract_domain(self, url: str) -> str:
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc
            return domain.replace('www.', '') if domain else ''
        except:
            return ''

    def _translate_response(self, response: Dict, target_lang: str) -> Dict:
        """Translate response"""
        translatable_fields = ['explanation', 'search_error']
        
        for field in translatable_fields:
            if field in response and response[field]:
                response[field] = translate_text(response[field], source='en', target=target_lang)
        
        if target_lang == 'hi' and 'label' in response:
            label_map = {'REAL': 'सच', 'FAKE': 'झूठ', 'UNVERIFIED': 'अप्रमाणित'}
            response['label_translated'] = label_map.get(response['label'], response['label'])
        
        return response

    def _get_inference_options(self) -> dict:
        if self.use_gpu:
            return {
                "temperature": 0.05,
                "num_predict": 500,
                "top_p": 0.85,
                "top_k": 30,
                "num_ctx": 8192,
                "num_gpu": self.gpu_layers,
                "num_thread": 8,
                "repeat_penalty": 1.15
            }
        return self._get_cpu_only_options()

    def _get_cpu_only_options(self) -> dict:
        return {
            "temperature": 0.05,
            "num_predict": 400,
            "top_p": 0.85,
            "top_k": 30,
            "num_ctx": 8192,
            "num_gpu": 80,
            "num_thread": 2,
            "repeat_penalty": 1.15
        }

    def _post_ollama(self, prompt: str, options: dict) -> requests.Response:
        return requests.post(
            self.ollama_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options
            },
            timeout=600
        )

    def _error_response(self, msg: str, ts: str) -> dict:
        return {
            "label": "UNVERIFIED",
            "confidence": 0.5,
            "explanation": f"Analysis error: {msg}. Unable to verify.",
            "sources": [],
            "timestamp": ts
        }


_verifier = None

def get_verifier():
    global _verifier
    if _verifier is None:
        logger.info("Initializing Enhanced Fact-Checker...")
        _verifier = EnhancedFactChecker(config=app_config)
    return _verifier