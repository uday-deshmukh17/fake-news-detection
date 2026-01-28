import json
from typing import List, Dict, Any


# --- HELPER FUNCTION TO FORMAT WEB RESULTS ---
def _format_web_context(web_context: List[Dict[str, Any]]) -> str:
    """
    Converts the list of search results into a clean,
    readable string for the model.
    """
    if not web_context:
        return "No web search results available. You must make your best judgment based on general knowledge and common sense."

    context_str = "AVAILABLE EVIDENCE FROM WEB SEARCH:\n\n"
    for i, source in enumerate(web_context, 1):
        title = source.get('title', 'No Title')
        snippet = source.get('snippet', 'No Snippet')
        url = source.get('url', 'No URL')

        context_str += f"[Source {i}]\n"
        context_str += f"Title: {title}\n"
        context_str += f"Content: {snippet}\n"
        context_str += f"URL: {url}\n\n"

    return context_str.strip()


# --- IMPROVED FEW-SHOT EXAMPLES ---
FEW_SHOT_EXAMPLES = [
    {
        "text": "NASA announces discovery of water on Mars",
        "web_sources": "Source [1] - NASA.gov confirms liquid water found in Mars polar ice caps. Source [2] - Scientific journal Nature publishes peer-reviewed study.",
        "output": {
            "label": "REAL",
            "confidence": 0.95,
            "explanation": "NASA's official website (Source [1]) and peer-reviewed scientific journal (Source [2]) both confirm the discovery of water on Mars with substantial evidence."
        }
    },
    {
        "text": "President signs law banning all social media",
        "web_sources": "Source [1] - Satirical news site 'TheOnion.com' publishes article. Source [2] - No official government sources found. Source [3] - Fact-checkers label as false.",
        "output": {
            "label": "FAKE",
            "confidence": 0.98,
            "explanation": "The only source is a satirical website (Source [1]), with no official government confirmation. Fact-checkers (Source [3]) explicitly labeled this as false."
        }
    },
    {
        "text": "Local restaurant wins best food award",
        "web_sources": "Source [1] - Local news website confirms award ceremony. Source [2] - Restaurant's official page announces the win.",
        "output": {
            "label": "REAL",
            "confidence": 0.85,
            "explanation": "Local news outlet (Source [1]) and the restaurant's official announcement (Source [2]) both confirm the award, though this is a local event with limited verification."
        }
    },
    {
        "text": "Aliens land in New York City, government confirms",
        "web_sources": "No credible sources found. Only tabloid and conspiracy websites mention this.",
        "output": {
            "label": "FAKE",
            "confidence": 0.99,
            "explanation": "No credible news sources, government statements, or official confirmations exist. Only unreliable tabloid websites mention this claim, which is scientifically implausible."
        }
    }
]

# --- IMPROVED PROMPT TEMPLATE ---
PROMPT_TEMPLATE = '''You are an expert fact-checker AI. Your job is to verify if news is REAL or FAKE.

{web_context}

INSTRUCTIONS:
1. Carefully read the web search results above
2. Check if credible sources (official sites, reputable news, verified accounts) confirm the claim
3. Consider source reliability: .gov, .edu, major news outlets are most reliable
4. Look for red flags: sensational language, no sources, conspiracy sites, satire websites
5. Be confident in your judgment - use high confidence (0.8-1.0) when evidence is clear

CONFIDENCE SCORING GUIDE:
- 0.9-1.0: Multiple credible sources confirm OR clear evidence of fake (satire sites, fact-checkers debunk)
- 0.7-0.89: At least one credible source confirms OR clear red flags indicate fake
- 0.5-0.69: Mixed evidence or uncertain sources
- Below 0.5: Insufficient information (avoid this - make a decision!)

FEW-SHOT EXAMPLES:
{few_shot_examples}

NOW EVALUATE THIS NEWS:
News: "{news_text}"

RESPOND WITH JSON ONLY (no extra text):
{{
  "label": "REAL" or "FAKE",
  "confidence": 0.0 to 1.0,
  "explanation": "One clear sentence citing sources by number, e.g., 'Source [1] from NASA.gov confirms...' or 'No credible sources found, only tabloid sites mention this claim.'"
}}

JSON Output:'''.strip()


# --- MODIFIED BUILD_PROMPT FUNCTION ---
def build_prompt(
        news_text: str,
        web_context: List[Dict[str, Any]] = None,
        use_few_shot: bool = True
) -> str:
    """
    Build an improved prompt with better instructions and examples
    """
    # 1. Format the web context
    formatted_context = _format_web_context(web_context)

    # 2. Format the few-shot examples with proper structure
    few_shot_str = ''
    if use_few_shot:
        few_shot_list = []
        for example in FEW_SHOT_EXAMPLES:
            few_shot_list.append({
                "news": example["text"],
                "sources_found": example["web_sources"],
                "your_response": example["output"]
            })
        few_shot_str = json.dumps(few_shot_list, indent=2)

    # 3. Format the final prompt
    return PROMPT_TEMPLATE.format(
        web_context=formatted_context,
        few_shot_examples=few_shot_str,
        news_text=news_text
    )