# VeriPulse Deployment Guide

## 🎯 Deployment Options

### Option 1: Docker (Recommended for Demo)

**Pros**: Easy setup, isolated environment, works anywhere
**Time**: 10 minutes

```bash
# 1. Clone and setup
git clone <repo-url>
cd veripulse

# 2. Configure environment
cp .env.template .env
# Edit .env with your API keys

# 3. Start services
docker-compose up -d

# 4. Check logs
docker-compose logs -f backend

# 5. Access
# Frontend: http://localhost:3000
# Backend: http://localhost:5000
# MongoDB: localhost:27017
```

---

### Option 2: Cloud Deployment (Production)

#### Backend (Railway)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo>
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to https://railway.app/
   - Click "New Project" → "Deploy from GitHub"
   - Select your repository
   - Add environment variables (all from `.env`)
   - Railway will auto-detect Flask and deploy

3. **Add MongoDB**
   - In Railway project, click "New"
   - Select "Database" → "MongoDB"
   - Copy connection string to `MONGODB_URI`

4. **Get Backend URL**
   - Railway provides URL like: `https://veripulse-backend-production.up.railway.app`

#### Frontend (Vercel)

1. **Deploy to Vercel**
   ```bash
   cd frontend
   vercel
   ```
   
2. **Or via Vercel Dashboard**
   - Go to https://vercel.com/
   - Click "Import Project"
   - Select your GitHub repo
   - Root directory: `frontend`
   - Framework: Create React App
   
3. **Add Environment Variable**
   - In Vercel project settings
   - Add: `REACT_APP_API_URL=<your-railway-backend-url>`
   
4. **Deploy**
   - Vercel deploys automatically on push to main

#### MongoDB Atlas (Database)

1. **Create Cluster**
   - Go to https://www.mongodb.com/cloud/atlas
   - Create free cluster
   - Whitelist all IPs (0.0.0.0/0) for demo
   
2. **Get Connection String**
   - Click "Connect" → "Connect your application"
   - Copy connection string
   - Replace `<password>` with your password
   
3. **Update Environment**
   - Update `MONGODB_URI` in Railway and local `.env`

---

### Option 3: Local Development

**Best for**: Active development during hackathon

```bash
# Terminal 1: MongoDB
mongod --dbpath ./data/db

# Terminal 2: Backend
cd backend
python app.py

# Terminal 3: Frontend
cd frontend
npm start

# Terminal 4: Bot Monitor (Optional)
cd backend
python bot_monitor.py
```

---

## 🔧 Environment Configuration

### Required API Keys

1. **OpenAI API Key**
   - Get from: https://platform.openai.com/api-keys
   - Used for: GPT-4 verification and explanations
   - Cost: ~$0.01 per verification

2. **Google Gemini API Key**
   - Get from: https://makersuite.google.com/app/apikey
   - Used for: Fast claim extraction
   - Free tier: 60 requests/minute

3. **Pinecone API Key**
   - Get from: https://www.pinecone.io/
   - Used for: Vector database
   - Free tier: 1 index, 100K vectors

### Optional (for Bot Functionality)

4. **Twitter/X API**
   - Apply at: https://developer.twitter.com/
   - Approval time: 1-2 days (plan ahead!)
   - Used for: Bot mentions processing

5. **Reddit API**
   - Create app: https://www.reddit.com/prefs/apps
   - Instant approval
   - Used for: Reddit bot

---

## 📊 Pre-Deployment Checklist

### Before Demo

- [ ] All API keys added to `.env`
- [ ] Pinecone index created (dimension: 768)
- [ ] MongoDB running and accessible
- [ ] Backend starts without errors
- [ ] Frontend connects to backend
- [ ] Test verification with sample claim
- [ ] Check that sources are being crawled
- [ ] Verify bot mentions work (if enabled)

### Testing Commands

```bash
# Test backend health
curl http://localhost:5000/health

# Test verification
curl -X POST http://localhost:5000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Tsunami warning issued in Mumbai"}'

# Test crawling
curl -X POST http://localhost:5000/api/crawl

# Check stats
curl http://localhost:5000/api/stats

# Test trending
curl http://localhost:5000/api/trending
```

---

## 🚀 Deployment Strategies for Hackathon

### Strategy 1: Demo-Ready (Recommended)

**Setup Time**: 2 hours
**Reliability**: High
**Features**: Full system

1. Deploy backend on Railway
2. Deploy frontend on Vercel
3. Use MongoDB Atlas
4. Pre-populate with sample verifications
5. Have backup local setup

### Strategy 2: Local-Only (Fallback)

**Setup Time**: 30 minutes
**Reliability**: Medium (network dependent)
**Features**: Full system

1. Run everything on your laptop
2. Use `localhost` URLs
3. Ensure stable WiFi
4. Have mobile hotspot backup

### Strategy 3: Hybrid

**Setup Time**: 1 hour
**Reliability**: Very High
**Features**: Full system + redundancy

1. Cloud deployment (primary)
2. Local setup (backup)
3. Can switch if cloud has issues
4. Show both in presentation

---

## 🎤 Demo Preparation

### Pre-load Sample Data

```python
# Run this script to pre-populate database
from agents.orchestrator_agent import OrchestratorAgent

orchestrator = OrchestratorAgent()

# Crawl sources
orchestrator.crawl_and_index_sources()

# Create sample verifications
sample_claims = [
    "Tsunami warning issued in Mumbai",
    "WHO declares new pandemic variant",
    "Earthquake reported in Delhi NCR magnitude 7.2"
]

for claim in sample_claims:
    result = orchestrator.verify_single_claim(claim)
    print(f"Verified: {claim} → {result['verdict']}")
```

### Demo Flow

1. **Show Web Portal**
   - Open http://localhost:3000 or Vercel URL
   - Paste sample claim
   - Explain as it processes
   - Show verdict, confidence, explanation
   - Highlight sources

2. **Show Bot Integration**
   - Show Twitter/Reddit mention
   - Explain how bot detects mentions
   - Show bot reply with verdict

3. **Show Trending**
   - Click "Trending" button
   - Explain real-time monitoring
   - Show clustered claims

4. **Technical Deep-Dive** (if time)
   - Show multi-agent architecture diagram
   - Explain RAG pipeline
   - Show vector database stats
   - Demonstrate real-time crawling

---

## 🐛 Common Deployment Issues

### Issue: "Pinecone index not found"

**Solution**:
```python
# Create index manually
import pinecone
pinecone.init(api_key="your-key", environment="gcp-starter")
pinecone.create_index("veripulse-evidence", dimension=768, metric="cosine")
```

### Issue: "MongoDB connection timeout"

**Solution**:
- Check MongoDB is running: `mongosh`
- Verify connection string in `.env`
- For Atlas: whitelist IP address

### Issue: "OpenAI rate limit exceeded"

**Solution**:
- Add payment method to OpenAI account
- Use cached verifications
- Switch to Gemini for some agents

### Issue: "Frontend can't connect to backend"

**Solution**:
- Check CORS is enabled in Flask
- Verify `REACT_APP_API_URL` is correct
- Check backend is running on correct port

### Issue: "NLI model download fails"

**Solution**:
```bash
# Pre-download model
python -c "from transformers import pipeline; pipeline('text-classification', model='microsoft/deberta-v3-base-mnli')"
```

---

## 📱 Mobile/Tablet Demo

If presenting on mobile:

1. Deploy both backend and frontend to cloud
2. Access Vercel frontend URL on mobile
3. Works fully responsive
4. Can show bot on Twitter/Reddit app

---

## 🎯 Performance Optimization

### For Demo

1. **Pre-cache Verifications**
   - Verify common claims beforehand
   - Store in MongoDB
   - Instant results during demo

2. **Warm Up Services**
   - Start all services 10 minutes before
   - Run test verifications
   - Ensure models are loaded

3. **Network Backup**
   - Use mobile hotspot as backup
   - Download models beforehand
   - Have offline slides

---

## 🏆 Judging Day Checklist

### 1 Day Before

- [ ] Deploy to cloud
- [ ] Test end-to-end flow
- [ ] Pre-populate sample data
- [ ] Prepare demo script
- [ ] Create backup plan

### Morning Of

- [ ] Verify all services running
- [ ] Test on venue WiFi
- [ ] Check API quota remaining
- [ ] Have MongoDB dump ready
- [ ] Charge laptop fully

### During Presentation

- [ ] Close unnecessary tabs/apps
- [ ] Use incognito mode (clean browser)
- [ ] Have backup verifications ready
- [ ] Monitor backend logs
- [ ] Stay calm if issues arise!

---

## 🎉 Post-Hackathon

### Cleanup

```bash
# Stop all services
docker-compose down

# Or stop individual services
# Railway: Pause project
# Vercel: Keep running (free)
# Pinecone: Keep free tier
```

### Share Project

- GitHub: Make repo public
- Demo video: Record walkthrough
- Documentation: This README!
- LinkedIn: Post about experience

---

**Good luck! 🚀**