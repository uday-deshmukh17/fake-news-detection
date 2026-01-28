from flask import Flask, request, jsonify
from classifier import get_verifier
import logging
import json
from datetime import datetime
import requests

app = Flask(__name__)

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Request counter for monitoring
request_count = 0
success_count = 0
error_count = 0

def validate_category_match(text, category):
    """
    Use Ollama Llama3.2 to validate if text matches the given category
    Returns: (is_valid, detected_category, message)
    """
    if category.lower() == "all":
        return True, "all", "Category validation skipped for 'all'"
    
    try:
        # Prepare improved prompt for Llama3.2
        prompt = f"""You are a news category classifier. Analyze the text and determine ONLY which category it belongs to.

Text to analyze: "{text}"

Available categories:
1. finance - Stock markets, cryptocurrencies, company valuations, business deals, economic indicators, banking, investments, GDP, inflation, interest rates, company earnings
2. entertainment - Movies, TV shows, music, celebrities, actors, singers, awards ceremonies, box office, streaming services, concerts, albums, Hollywood, Bollywood
3. sports - Football, cricket, basketball, tennis, Olympics, tournaments, matches, players, transfers, records, championships, leagues, athletes

IMPORTANT RULES:
- Only respond with ONE word: "finance" OR "entertainment" OR "sports"
- Be strict: Only classify as the requested category if it clearly matches
- Focus on the MAIN topic, not peripheral mentions
- If text mentions money/prices about entertainment (like movie earnings), it's still "entertainment"
- If text mentions money/prices about sports (like player salaries), it's still "sports"

Examples:
- "Bitcoin reaches $150,000" → finance
- "Movie earns $1 billion at box office" → entertainment
- "Football player signs $50 million contract" → sports
- "Taylor Swift releases new album" → entertainment
- "India wins Cricket World Cup" → sports
- "Tesla stock splits 10-to-1" → finance

Analyze the text above and respond with ONLY ONE WORD (finance/entertainment/sports):"""

        # Call Ollama API
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Very low for consistent results
                    "top_p": 0.9,
                    "num_predict": 10  # Only need one word
                }
            },
            timeout=15
        )
        
        if ollama_response.status_code != 200:
            logger.warning(f"Ollama API returned status {ollama_response.status_code}")
            return True, category, "Category validation failed, proceeding anyway"
        
        # Extract detected category
        response_data = ollama_response.json()
        detected_text = response_data.get("response", "").strip().lower()
        
        # Extract just the category word
        detected_category = None
        for cat in ["finance", "entertainment", "sports"]:
            if cat in detected_text:
                detected_category = cat
                break
        
        if not detected_category:
            logger.warning(f"Could not determine category from Ollama response: {detected_text}")
            return True, category, "Category validation inconclusive, proceeding anyway"
        
        logger.info(f"Category validation: Expected='{category}', Detected='{detected_category}'")
        
        # Check if they match
        if detected_category == category.lower():
            return True, detected_category, f"Text matches '{category}' category"
        else:
            return False, detected_category, f"Please ask the question about {detected_category}"
            
    except requests.exceptions.Timeout:
        logger.warning("Ollama API timeout during category validation")
        return True, category, "Category validation timeout, proceeding anyway"
    except Exception as e:
        logger.error(f"Category validation error: {e}")
        return True, category, f"Category validation error, proceeding anyway"

@app.errorhandler(500)
def internal_error(error):
    global error_count
    error_count += 1
    logger.error(f"Internal error: {error}")
    return jsonify({
        "error": "Internal Server Error",
        "detail": str(error),
        "timestamp": datetime.now().isoformat()
    }), 500

@app.errorhandler(400)
def bad_request(error):
    global error_count
    error_count += 1
    return jsonify({
        "error": "Bad Request",
        "detail": str(error),
        "timestamp": datetime.now().isoformat()
    }), 400

@app.route("/classify", methods=["POST"])
def classify_item():
    """Enhanced classification endpoint with comprehensive validation"""
    global request_count, success_count, error_count
    request_count += 1
    
    start_time = datetime.now()
    logger.info(f"[Request #{request_count}] Classification request received")
    
    # Validate request
    if not request.json:
        logger.warning("Missing JSON body")
        return jsonify({"error": "Request must be JSON"}), 400
    
    if "text" not in request.json:
        logger.warning("Missing 'text' field")
        return jsonify({"error": "Missing 'text' in request body"}), 400

    news_item = request.json
    text_to_classify = news_item.get("text")
    category = news_item.get("category", "all")
    language = news_item.get("language", "en")
    item_id = news_item.get("id", f"req_{request_count}")

    # Validate text
    if not text_to_classify or len(text_to_classify.strip()) == 0:
        logger.warning("Empty text provided")
        return jsonify({"error": "Text cannot be empty"}), 400
    
    if len(text_to_classify) > 1000:
        logger.warning(f"Text too long: {len(text_to_classify)} chars")
        return jsonify({
            "error": "Text too long",
            "detail": "Maximum 1000 characters allowed",
            "length": len(text_to_classify)
        }), 400

    # Validate category
    valid_categories = ["all", "finance", "entertainment", "sports"]
    if category.lower() not in valid_categories:
        logger.warning(f"Invalid category: {category}")
        return jsonify({
            "error": "Invalid category",
            "detail": f"Must be one of: {', '.join(valid_categories)}",
            "provided": category
        }), 400

    # Validate language
    valid_languages = ["en", "hi"]
    if language.lower() not in valid_languages:
        logger.warning(f"Invalid language: {language}")
        return jsonify({
            "error": "Invalid language",
            "detail": f"Must be one of: {', '.join(valid_languages)}",
            "provided": language
        }), 400

    # ⭐ NEW: Validate category match using Ollama ⭐
    is_valid, detected_category, validation_message = validate_category_match(text_to_classify, category)
    
    if not is_valid:
        logger.warning(f"[Request #{request_count}] Category mismatch: {validation_message}")
        error_count += 1
        return jsonify({
            "error": "Category mismatch",
            "detail": validation_message,
            "detected_category": detected_category,
            "requested_category": category,
            "timestamp": datetime.now().isoformat()
        }), 400

    logger.info(f"[Request #{request_count}] Text: '{text_to_classify[:50]}...', Category: {category}, Language: {language}")
    logger.info(f"[Request #{request_count}] Category validation: {validation_message}")

    try:
        # Get verifier
        verifier = get_verifier()

        # Perform classification
        result = verifier.classify(text_to_classify, category, language)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"[Request #{request_count}] Classification complete: {result.get('label')} (confidence: {result.get('confidence', 0):.2f}) in {processing_time:.2f}s")

        # ⭐ Save to JSONL for history
        try:
            result_record = {
                "request_id": item_id,
                "timestamp": start_time.isoformat(),
                "input": text_to_classify,
                "category": category,
                "detected_category": detected_category,
                "language": language,
                "result": result,
                "processing_time": processing_time
            }
            
            with open("classification_history.jsonl", "a", encoding='utf-8') as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save classification history: {e}")

        # ⭐ ALSO Save latest result to result.json ⭐
        try:
            with open("result.json", "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("✅ Result saved to result.json")
        except Exception as e:
            logger.warning(f"Failed to save to result.json: {e}")

        success_count += 1

        # Build response
        response = {
            "id": item_id,
            "request_number": request_count,
            "input": text_to_classify,
            "category": category,
            "language": language,
            "result": result,
            "processing_time": processing_time,
            "timestamp": start_time.isoformat()
        }

        return jsonify(response)

    except Exception as e:
        error_count += 1
        logger.error(f"[Request #{request_count}] Classification failed: {str(e)}", exc_info=True)
        
        return jsonify({
            "error": "Classification failed",
            "detail": str(e)[:200],
            "id": item_id,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Enhanced health check endpoint"""
    try:
        # Check Ollama connection
        ollama_status = "unknown"
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            ollama_status = "healthy" if resp.status_code == 200 else "unhealthy"
        except:
            ollama_status = "unreachable"
        
        # System stats
        uptime = datetime.now().isoformat()
        
        return jsonify({
            "status": "healthy" if ollama_status == "healthy" else "degraded",
            "ollama": ollama_status,
            "model": "llama3.2",
            "mode": "gpu_cpu_hybrid",
            "supported_languages": ["en", "hi"],
            "supported_categories": ["all", "finance", "entertainment", "sports"],
            "statistics": {
                "total_requests": request_count,
                "successful": success_count,
                "errors": error_count,
                "success_rate": f"{(success_count/request_count*100):.1f}%" if request_count > 0 else "N/A"
            },
            "timestamp": uptime
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/categories", methods=["GET"])
def get_categories():
    """Get supported categories with descriptions"""
    return jsonify({
        "supported_categories": [
            {
                "id": "all",
                "name": "All Categories",
                "description": "Search across all news categories. Best for general claims."
            },
            {
                "id": "finance",
                "name": "Finance & Business",
                "description": "Financial news, market data, business announcements, company valuations."
            },
            {
                "id": "entertainment",
                "name": "Entertainment & Celebrity",
                "description": "Movies, music, celebrities, awards, entertainment industry news."
            },
            {
                "id": "sports",
                "name": "Sports",
                "description": "Sports events, matches, tournaments, player transfers, records."
            }
        ],
        "default_category": "all",
        "note": "Category helps find more relevant sources but system can verify claims in any category."
    })

@app.route("/languages", methods=["GET"])
def get_languages():
    """Get supported languages"""
    return jsonify({
        "supported_languages": [
            {
                "code": "en",
                "name": "English",
                "native_name": "English"
            },
            {
                "code": "hi",
                "name": "Hindi",
                "native_name": "हिंदी"
            }
        ],
        "default_language": "en",
        "note": "System translates input to English for analysis, then translates results back."
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get server statistics"""
    try:
        # Read recent classifications
        recent_classifications = []
        try:
            with open("classification_history.jsonl", "r", encoding='utf-8') as f:
                lines = f.readlines()
                recent_classifications = [json.loads(line) for line in lines[-10:]]  # Last 10
        except:
            pass
        
        # Calculate stats
        label_counts = {"REAL": 0, "FAKE": 0, "UNVERIFIED": 0}
        category_counts = {}
        avg_confidence = 0
        avg_processing_time = 0
        
        if recent_classifications:
            for record in recent_classifications:
                label = record.get('result', {}).get('label', 'UNKNOWN')
                label_counts[label] = label_counts.get(label, 0) + 1
                
                category = record.get('category', 'all')
                category_counts[category] = category_counts.get(category, 0) + 1
                
                avg_confidence += record.get('result', {}).get('confidence', 0)
                avg_processing_time += record.get('processing_time', 0)
            
            avg_confidence /= len(recent_classifications)
            avg_processing_time /= len(recent_classifications)
        
        return jsonify({
            "total_requests": request_count,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "success_rate": f"{(success_count/request_count*100):.1f}%" if request_count > 0 else "N/A",
            "recent_labels": label_counts,
            "recent_categories": category_counts,
            "average_confidence": round(avg_confidence, 2),
            "average_processing_time": round(avg_processing_time, 2),
            "recent_classifications": len(recent_classifications),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return jsonify({
            "error": "Failed to retrieve statistics",
            "detail": str(e)
        }), 500

@app.route("/", methods=["GET"])
def index():
    """API information page"""
    return jsonify({
        "name": "Enhanced Fake News Detector API",
        "version": "2.0",
        "description": "Production-ready fake news detection with multi-layer verification",
        "endpoints": {
            "POST /classify": "Classify news text as REAL, FAKE, or UNVERIFIED",
            "GET /health": "Check server health status",
            "GET /categories": "List supported categories",
            "GET /languages": "List supported languages",
            "GET /stats": "View server statistics",
            "GET /": "API information"
        },
        "features": [
            "Multi-layer verification (6 algorithms)",
            "Advanced web scraping (Selenium + trafilatura)",
            "ML-based credibility scoring",
            "Numerical claim verification",
            "Entity extraction and matching",
            "Semantic analysis",
            "Multi-language support (EN, HI)",
            "Category-specific search",
            "AI-powered category validation"
        ],
        "documentation": "See README.md for complete documentation",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ENHANCED FAKE NEWS DETECTOR SERVER")
    logger.info("=" * 80)
    logger.info("Starting server...")
    logger.info("API Endpoints:")
    logger.info("  POST /classify    - Classify news")
    logger.info("  GET  /health      - Health check")
    logger.info("  GET  /categories  - List categories")
    logger.info("  GET  /languages   - List languages")
    logger.info("  GET  /stats       - View statistics")
    logger.info("  GET  /            - API info")
    logger.info("=" * 80)
    
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)