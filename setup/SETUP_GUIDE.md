# 🛡️ TruthGuard - Complete Setup Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [API Documentation](#api-documentation)
6. [Advanced Features](#advanced-features)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Anthropic API Key
- Docker (optional, for containerized deployment)
- 4GB RAM minimum (8GB recommended)

### Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/truthguard.git
cd truthguard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python -m textblob.download_corpora

# Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run in 30 seconds

```bash
# Start the server
python main.py

# In another terminal, test the system
curl -X POST http://localhost:8000/api/init \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-anthropic-key"}'

# Submit a test claim
curl -X POST http://localhost:8000/api/claims/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news: Major policy change announced"}'
```

---

## 📦 Installation Details

### Option 1: Local Installation

```bash
# 1. System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3.11 python3-pip python3-venv git

# 2. Clone and setup
git clone https://github.com/yourusername/truthguard.git
cd truthguard
python3.11 -m venv venv
source venv/bin/activate

# 3. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download ML models
python -m textblob.download_corpora
```

### Option 2: Docker Installation

```bash
# 1. Build image
docker build -t truthguard:latest .

# 2. Run container
docker-compose up -d

# 3. Check status
docker-compose ps
docker-compose logs -f truthguard-api
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

```bash
# See detailed deployment section below
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Optional
MONGODB_URI=mongodb://localhost:27017/truthguard  # For persistence
REDIS_URL=redis://localhost:6379/0                # For caching
LOG_LEVEL=INFO                                     # DEBUG, INFO, WARNING, ERROR
MAX_CONCURRENT_VERIFICATIONS=10                    # Parallel processing limit
```

### Advanced Configuration

Create `config.json`:

```json
{
  "agents": {
    "scanner": {
      "scan_interval_seconds": 30,
      "sources": ["social_media", "news_outlets", "forums"]
    },
    "verifier": {
      "timeout_seconds": 120,
      "max_sources": 20,
      "min_credibility_score": 0.6
    },
    "analyzer": {
      "enable_impact_prediction": true,
      "enable_network_analysis": true
    }
  },
  "features": {
    "temporal_tracking": true,
    "multimodal_analysis": true,
    "predictive_modeling": true
  }
}
```

---

## 🏃 Running the System

### Development Mode

```bash
# Start with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the built-in runner
python main.py
```

### Production Mode

```bash
# Using Gunicorn with Uvicorn workers
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Background Monitoring

```bash
# Start continuous monitoring
python -c "
import asyncio
from main import orchestrator

async def main():
    await orchestrator.continuous_monitoring()

asyncio.run(main())
"
```

---

## 📚 API Documentation

### Initialize System

```bash
POST /api/init
Content-Type: application/json

{
  "api_key": "your-anthropic-api-key"
}

Response:
{
  "status": "initialized",
  "message": "System ready"
}
```

### Submit Claim for Verification

```bash
POST /api/claims/submit
Content-Type: application/json

{
  "text": "Claim text to verify"
}

Response:
{
  "claim_id": "abc123def456",
  "status": "processing",
  "message": "Claim submitted for verification"
}
```

### Get Claim Details

```bash
GET /api/claims/{claim_id}

Response:
{
  "id": "abc123def456",
  "text": "Claim text",
  "status": "verified|false|misleading|unverifiable",
  "confidence": 0.85,
  "severity": "critical|high|medium|low",
  "viral_score": 0.75,
  "detected_at": "2025-01-15T10:30:00",
  "verified_at": "2025-01-15T10:33:12",
  "summary": "Verification summary",
  "entities": ["entity1", "entity2"],
  "context": {
    "historical_background": "...",
    "related_facts": ["fact1", "fact2"]
  },
  "public_alert": "Public-friendly explanation"
}
```

### List Recent Claims

```bash
GET /api/claims?limit=20&status=false

Response:
{
  "claims": [
    {
      "id": "abc123",
      "text": "Claim text",
      "status": "false",
      "confidence": 0.92,
      "severity": "high",
      "detected_at": "2025-01-15T10:30:00",
      "sources_count": 12
    }
  ],
  "total": 156
}
```

### Get System Statistics

```bash
GET /api/stats

Response:
{
  "claims_scanned": 1247,
  "claims_verified": 1190,
  "misinformation_detected": 89,
  "sources_monitored": 156,
  "avg_verify_time": 3.2
}
```

### WebSocket Live Feed

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Live update:', data);
  // data.claims - recent claims
  // data.stats - current statistics
};
```

---

## 🔥 Advanced Features

### 1. Temporal Claim Tracking

Tracks how claims evolve and mutate over time:

```python
from advanced_features import TemporalTracker

tracker = TemporalTracker(client)
evolution = await tracker.track_claim_evolution(
    original_claim="Original claim text",
    new_variant="Modified claim text"
)

print(f"Same claim: {evolution['is_same_claim']}")
print(f"Similarity: {evolution['similarity_score']}")
print(f"Mutations: {evolution['mutations']}")
```

### 2. Network Analysis

Detects coordinated disinformation campaigns:

```python
from advanced_features import NetworkAnalyzer

analyzer = NetworkAnalyzer(client)
campaign = await analyzer.detect_coordinated_campaign(
    claim_id="abc123",
    sources=source_list
)

if campaign['is_coordinated']:
    print(f"Coordinated campaign detected!")
    print(f"Score: {campaign['campaign_score']}")
    print(f"Indicators: {campaign['indicators']}")
```

### 3. Predictive Modeling

Predicts future spread and impact:

```python
from advanced_features import PredictiveModeler

modeler = PredictiveModeler(client)
prediction = await modeler.predict_spread(claim, current_metrics)

print(f"Predicted peak reach: {prediction['predicted_peak_reach']}")
print(f"Spread velocity: {prediction['spread_velocity']} sources/hour")
print(f"Longevity: {prediction['longevity_days']} days")
```

### 4. Multi-Modal Analysis

Analyzes images for manipulation:

```python
from advanced_features import MultiModalAnalyzer

analyzer = MultiModalAnalyzer(client)
result = await analyzer.analyze_image(image_bytes, claim_text)

if result['manipulation_detected']:
    print(f"Manipulation found! Confidence: {result['manipulation_confidence']}")
    print(f"Signs: {result['manipulation_signs']}")
```

### 5. Adaptive Explanations

Generates audience-specific explanations:

```python
from advanced_features import ExplanationEngine

engine = ExplanationEngine(client)

# For general public
general_explanation = await engine.generate_explanation(
    claim, verification, audience_level="general"
)

# For technical audience
technical_explanation = await engine.generate_explanation(
    claim, verification, audience_level="technical"
)

# Counter-narrative
counter = await engine.generate_counter_narrative(
    false_claim="False claim text",
    facts=["fact1", "fact2", "fact3"]
)
```

---

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agents.py -v

# Run with coverage
pytest tests/ --cov=main --cov-report=html
```

### Integration Tests

```bash
# Test complete workflow
pytest tests/test_integration.py -v

# Test API endpoints
pytest tests/test_api.py -v
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

### Example Test

```python
# tests/test_verification.py
import pytest
from main import AgentOrchestrator

@pytest.mark.asyncio
async def test_claim_verification():
    orchestrator = AgentOrchestrator("test-api-key")
    
    claim = await orchestrator.process_claim(
        "Test claim for verification"
    )
    
    assert claim.id is not None
    assert claim.status in ["verified", "false", "misleading"]
    assert 0 <= claim.confidence <= 1
```

---

## 🚀 Deployment

### AWS Deployment

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name truthguard

# 2. Build and push image
docker build -t truthguard:latest .
docker tag truthguard:latest {account-id}.dkr.ecr.{region}.amazonaws.com/truthguard:latest
docker push {account-id}.dkr.ecr.{region}.amazonaws.com/truthguard:latest

# 3. Deploy to ECS
aws ecs create-cluster --cluster-name truthguard-cluster
aws ecs create-service --cli-input-json file://ecs-service.json
```

### Google Cloud Platform

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/{project-id}/truthguard

# 2. Deploy to Cloud Run
gcloud run deploy truthguard \
  --image gcr.io/{project-id}/truthguard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ANTHROPIC_API_KEY=xxx
```

### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: truthguard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: truthguard
  template:
    metadata:
      labels:
        app: truthguard
    spec:
      containers:
      - name: truthguard
        image: truthguard:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: truthguard-secrets
              key: api-key
```

```bash
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. API Key Errors

```
Error: Invalid API key

Solution:
- Check .env file has correct ANTHROPIC_API_KEY
- Ensure no extra spaces or quotes
- Verify key is active at console.anthropic.com
```

#### 2. Slow Verification

```
Issue: Claims taking >5 minutes to verify

Solutions:
- Increase MAX_CONCURRENT_VERIFICATIONS
- Check internet connection
- Monitor API rate limits
- Consider caching results
```

#### 3. Memory Issues

```
Error: Out of memory

Solutions:
- Increase Docker memory limit
- Reduce MAX_CONCURRENT_VERIFICATIONS
- Enable result caching with Redis
- Implement claim queue system
```

#### 4. WebSocket Connection Fails

```
Error: WebSocket connection refused

Solutions:
- Check firewall settings
- Verify port 8000 is open
- Check CORS configuration
- Use wss:// for HTTPS deployments
```

### Debug Mode

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python main.py

# Check logs
tail -f logs/truthguard.log

# Monitor system resources
htop
docker stats  # if using Docker
```

---

## 📊 Performance Optimization

### Caching Strategy

```python
# Enable Redis caching
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=3600
ENABLE_RESULT_CACHING=True
```

### Database Indexing

```python
# MongoDB indexes for fast queries
db.claims.createIndex({"detected_at": -1})
db.claims.createIndex({"status": 1, "confidence": -1})
db.claims.createIndex({"entities": 1})
```

### Load Balancing

```nginx
# nginx.conf
upstream truthguard_backend {
    least_conn;
    server app1:8000 weight=3;
    server app2:8000 weight=3;
    server app3:8000 weight=2;
}

server {
    listen 80;
    location / {
        proxy_pass http://truthguard_backend;
    }
}
```

---

## 📈 Monitoring & Alerts

### Prometheus Metrics

```python
# Add to main.py
from prometheus_client import Counter, Histogram

claims_processed = Counter('claims_processed_total', 'Total claims processed')
verification_duration = Histogram('verification_duration_seconds', 'Verification time')
```

### Health Checks

```bash
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /api/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

---

## 🎓 Best Practices

1. **Always use environment variables** for sensitive data
2. **Enable caching** for production deployments
3. **Monitor API usage** to avoid rate limits
4. **Implement retry logic** for failed verifications
5. **Use WebSockets** for real-time updates
6. **Regular backups** of claim database
7. **Rate limiting** for public APIs
8. **Security headers** in production

---

## 📞 Support

- **Documentation**: https://docs.truthguard.ai
- **GitHub Issues**: https://github.com/yourusername/truthguard/issues
- **Discord**: https://discord.gg/truthguard
- **Email**: support@truthguard.ai

---

## 📄 License

MIT License - See LICENSE file for details

---

**Happy Fact-Checking! 🎯**