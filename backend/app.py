# backend/app.py
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from agents.orchestrator_agent import OrchestratorAgent
from config import Config
import logging
from functools import wraps
import time

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - COMPLETE FIX
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
            "expose_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "max_age": 3600
        }
    }
)

# Additional CORS headers for all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS,PATCH')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate configuration
try:
    Config.validate()
    logger.info("✓ Configuration validated successfully")
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    logger.error("Please check your .env file and try again")
    exit(1)

# Initialize orchestrator
try:
    orchestrator = OrchestratorAgent()
    logger.info("✓ OrchestratorAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OrchestratorAgent: {e}")
    logger.error("The application will start but may not function correctly")
    orchestrator = None


# Rate limiting decorator
def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Simple in-memory rate limiting (use Redis in production)
        try:
            user_id = request.json.get('user_id') if request.json else None
            if user_id:
                # Check rate limit in MongoDB
                pass
        except:
            pass  # Ignore rate limit errors
        return f(*args, **kwargs)
    return decorated


# Timeout handler decorator
def with_timeout(timeout_seconds=120):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except RuntimeError as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    logger.error(f"Request timeout in {f.__name__}: {error_msg}")
                    return jsonify({
                        "error": "Request timeout",
                        "message": "The verification process took too long. The LLM server may be slow or overloaded. Please try again.",
                        "details": error_msg
                    }), 504
                raise
        return decorated
    return decorator


@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    return jsonify({
        "status": "healthy",
        "service": "VeriPulse API",
        "version": "1.0.0",
        "orchestrator_ready": orchestrator is not None,
        "cors": "enabled"
    })


@app.route('/api/verify', methods=['POST', 'OPTIONS'])
@rate_limit
@with_timeout(timeout_seconds=3000)
def verify_claim():
    """Verify a single claim (main endpoint)"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if orchestrator is None:
        return jsonify({
            "error": "Service unavailable",
            "message": "Orchestrator not initialized"
        }), 503
    
    try:
        data = request.json
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        user_id = data.get('user_id')
        
        if len(text) < 10:
            return jsonify({"error": "Claim text too short (minimum 10 characters)"}), 400
        
        if len(text) > 1000:
            return jsonify({"error": "Claim text too long (maximum 1000 characters)"}), 400
        
        # Verify claim
        logger.info(f"Verifying claim: {text[:100]}...")
        result = orchestrator.verify_single_claim(text, user_id, platform="web")
        
        if "error" in result:
            return jsonify(result), 429
        
        return jsonify({
            "success": True,
            "data": result
        })
    
    except RuntimeError as e:
        # Let the timeout decorator handle it
        raise
    
    except Exception as e:
        logger.exception(f"Error verifying claim: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/api/verify-url', methods=['POST', 'OPTIONS'])
@rate_limit
@with_timeout(timeout_seconds=120)
def verify_url():
    """Verify a claim from a URL"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if orchestrator is None:
        return jsonify({
            "error": "Service unavailable",
            "message": "Orchestrator not initialized"
        }), 503
    
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({"error": "Missing 'url' field"}), 400
        
        url = data['url']
        
        # Fetch article content
        from services.source_crawler import SourceCrawler
        crawler = SourceCrawler()
        article = crawler.fetch_article_content(url)
        
        if not article:
            return jsonify({"error": "Could not fetch article"}), 400
        
        # Extract main claim from article
        text = f"{article['title']} {article['text'][:500]}"
        
        # Verify
        result = orchestrator.verify_single_claim(text, platform="web")
        result['source_url'] = url
        result['source_title'] = article['title']
        
        return jsonify({
            "success": True,
            "data": result
        })
    
    except Exception as e:
        logger.exception(f"Error verifying URL: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/trending', methods=['GET', 'OPTIONS'])
def get_trending():
    """Get trending verified claims"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if orchestrator is None:
        return jsonify({
            "error": "Service unavailable",
            "message": "Orchestrator not initialized"
        }), 503
    
    try:
        keywords = request.args.get('keywords', 'tsunami,earthquake,flood,pandemic').split(',')
        logger.info(f"Monitoring trending claims for keywords: {keywords}")
        
        trending = orchestrator.monitor_trending_claims(keywords)
        
        return jsonify({
            "success": True,
            "data": trending,
            "count": len(trending)
        })
    
    except Exception as e:
        logger.exception(f"Error getting trending: {e}")
        return jsonify({
            "error": str(e),
            "success": False,
            "data": [],
            "count": 0
        }), 500


@app.route('/api/bot/mentions', methods=['POST', 'OPTIONS'])
def process_mentions():
    """Process bot mentions (called by bot monitor service)"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if orchestrator is None:
        return jsonify({
            "error": "Service unavailable",
            "message": "Orchestrator not initialized"
        }), 503
    
    try:
        # Authenticate bot service (add auth token in production)
        results = orchestrator.process_bot_mentions_batch()
        
        return jsonify({
            "success": True,
            "processed": len(results),
            "results": results
        })
    
    except Exception as e:
        logger.exception(f"Error processing mentions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/crawl', methods=['POST', 'OPTIONS'])
def trigger_crawl():
    """Trigger source crawling (admin endpoint)"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if orchestrator is None:
        return jsonify({
            "error": "Service unavailable",
            "message": "Orchestrator not initialized"
        }), 503
    
    try:
        # Add authentication in production
        logger.info("Starting source crawl...")
        count = orchestrator.crawl_and_index_sources()
        
        return jsonify({
            "success": True,
            "message": f"Indexed {count} articles"
        })
    
    except Exception as e:
        logger.exception(f"Error crawling: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET', 'OPTIONS'])
def get_stats():
    """Get system statistics"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if orchestrator is None:
        return jsonify({
            "error": "Service unavailable",
            "message": "Orchestrator not initialized"
        }), 503
    
    try:
        stats = orchestrator.get_statistics()
        
        return jsonify({
            "success": True,
            "data": stats
        })
    
    except Exception as e:
        logger.exception(f"Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/history', methods=['GET', 'OPTIONS'])
def get_history():
    """Get verification history"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        limit = int(request.args.get('limit', 20))
        
        # Get recent verifications from MongoDB
        from services.mongodb_service import MongoDBService
        mongo = MongoDBService()
        verifications = list(mongo.db.verifications.find().sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId to string
        for v in verifications:
            v['_id'] = str(v['_id'])
            v['claim_id'] = str(v.get('claim_id', ''))
        
        return jsonify({
            "success": True,
            "data": verifications,
            "count": len(verifications)
        })
    
    except Exception as e:
        logger.exception(f"Error getting history: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(504)
def timeout_error(e):
    return jsonify({
        "error": "Request timeout",
        "message": "The request took too long to process"
    }), 504


if __name__ == '__main__':
    # Initial crawl on startup (optional, can be slow)
    if orchestrator:
        logger.info("Starting initial source crawl...")
        try:
            count = orchestrator.crawl_and_index_sources()
            logger.info(f"✓ Indexed {count} articles from trusted sources")
        except Exception as e:
            logger.error(f"Initial crawl failed: {e}")
    
    # Start Flask app
    logger.info("=" * 60)
    logger.info("VeriPulse API Server Starting...")
    logger.info("=" * 60)
    logger.info(f"Environment: {Config.FLASK_ENV}")
    logger.info(f"MongoDB: {Config.MONGODB_DB}")
    logger.info(f"LLM Host: {Config.LLM_HOST if hasattr(Config, 'LLM_HOST') else 'Not configured'}")
    logger.info(f"CORS: Enabled (All origins)")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.FLASK_ENV == 'development',
        threaded=True
    )