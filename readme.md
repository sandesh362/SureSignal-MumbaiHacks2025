# VeriPulse - AI Agent for Detecting and Verifying Misinformation

## 🎯 Overview

VeriPulse is an AI-powered verification system that detects, verifies, and explains misinformation in real-time using a collaborative multi-agent architecture with RAG (Retrieval-Augmented Generation).

### Key Features

- **6 Collaborative AI Agents** using CrewAI
- **RAG Pipeline** with Pinecone vector database + MongoDB
- **Real-time Source Crawling** from trusted sources (PIB, WHO, AP News)
- **NLP Entailment Models** for fact verification
- **Twitter/Reddit Bot** with `@VeriPulseBot` and `#VeriCheck`
- **Web Portal** for public verification
- **ELI12 Explanations** for accessibility

---

## 🏗️ Architecture

### Multi-Agent System

1. **Ingestion/Collector Agent** - Monitors Twitter, Reddit, RSS feeds
2. **Claim Extraction & Clustering Agent** - Extracts claims, deduplicates, detects trends
3. **Evidence Retrieval Agent** - RAG queries + real-time crawling
4. **Veracity Scoring Agent** - NLI entailment + confidence scoring
5. **Explanation Agent** - Generates detailed + ELI12 explanations
6. **Orchestrator Agent** - Coordinates workflow, rate limiting, bot replies

### Tech Stack

- **Backend**: Flask + CrewAI + LangChain
- **Frontend**: React + Tailwind CSS
- **LLMs**: OpenAI GPT-4 + Google Gemini
- **Databases**: MongoDB (metadata) + Pinecone (embeddings)
- **NLP**: Sentence Transformers, DeBERTa (NLI)
- **Deployment**: Docker + Docker Compose

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- Docker & Docker Compose
- OpenAI API key
- Google Gemini API key
- Pinecone account

### 1. Clone & Setup

```bash
# Clone repository
git clone <your-repo-url>
cd veripulse

# Create environment file
cp .env.template .env
# Edit .env with your API keys
```

### 2. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### 3. Setup Pinecone

1. Create account at https://www.pinecone.io/
2. Create a new index named `veripulse-evidence`
3. Use dimension: `768` (for sentence-transformers/all-mpnet-base-v2)
4. Metric: `cosine`
5. Add API key to `.env`

### 4. Run with Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- MongoDB: localhost:27017

### 5. Run Locally (Development)

**Terminal 1 - MongoDB:**
```bash
mongod --dbpath ./data/db
```

**Terminal 2 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm start
```

---

## 📡 API Endpoints

### Verify Text
```bash
POST /api/verify
Content-Type: application/json

{
  "text": "Tsunami warning issued in Mumbai",
  "user_id": "optional_user_id"
}
```

### Verify URL
```bash
POST /api/verify-url
Content-Type: application/json

{
  "url": "https://example.com/article"
}
```

### Get Trending Claims
```bash
GET /api/trending?keywords=tsunami,earthquake,flood
```

### System Statistics
```bash
GET /api/stats
```

---

## 🤖 Bot Setup (Optional)

### Twitter Bot

1. Create Twitter Developer account
2. Create app and get API keys
3. Add credentials to `.env`:
   ```
   TWITTER_API_KEY=...
   TWITTER_API_SECRET=...
   TWITTER_ACCESS_TOKEN=...
   TWITTER_ACCESS_SECRET=...
   TWITTER_BEARER_TOKEN=...
   ```

4. Users tag bot: `@VeriPulseBot #VeriCheck <claim>`

### Reddit Bot

1. Create Reddit app at https://www.reddit.com/prefs/apps
2. Get client ID and secret
3. Add to `.env`:
   ```
   REDDIT_CLIENT_ID=...
   REDDIT_CLIENT_SECRET=...
   ```

4. Users mention: `u/VeriPulseBot <claim>`

### Process Mentions

```bash
# Trigger bot to check mentions
POST /api/bot/mentions
```

---

## 🔍 How It Works

### Verification Pipeline

1. **User submits claim** (via bot tag or web portal)
2. **Claim extraction** - Canonicalize and normalize text
3. **Evidence retrieval**:
   - Search vector DB for similar evidence
   - If insufficient, crawl trusted sources in real-time
   - Index new evidence
4. **Veracity check**:
   - Run NLI entailment model
   - LLM-based semantic analysis
   - Aggregate into verdict: TRUE/FALSE/MISLEADING/UNVERIFIED
5. **Explanation generation**:
   - Detailed explanation
   - ELI12 simple version
   - Citations with sources
6. **Response delivery**:
   - Bot replies on social media
   - Web portal displays full report

### RAG System

- **Embedding Model**: sentence-transformers/all-mpnet-base-v2
- **Vector Store**: Pinecone (768 dimensions)
- **Similarity Search**: Top-5 most relevant evidence
- **Real-time Crawling**: Fetches fresh articles from PIB, WHO, AP News
- **Metadata Storage**: MongoDB (titles, URLs, timestamps)

---

## 📊 Example Usage

### Web Portal

1. Go to http://localhost:3000
2. Enter claim: "Tsunami warning issued in Mumbai"
3. Click "Verify Claim"
4. View results:
   - Verdict: FALSE
   - Confidence: 85%
   - Explanation: "No official tsunami warning from IMD or NDMA"
   - Sources: [IMD Report], [NDMA Update]

### Twitter Bot

```
Tweet: @VeriPulseBot #VeriCheck Tsunami warning in Mumbai!

Bot Reply:
❌ Verdict: FALSE (85% confidence)

📝 No official tsunami warning has been issued by IMD or NDMA 
for Mumbai. This claim is not supported by authoritative sources.

📚 Sources:
• IMD: No active tsunami alerts
  https://mausam.imd.gov.in/...
• NDMA: No coastal warnings
  https://ndma.gov.in/...

🔗 Full report: [link]
```

---

## 🔧 Configuration

### Trusted Sources

Edit `backend/config.py` to add/remove sources:

```python
TRUSTED_SOURCES = {
    "pib": "https://pib.gov.in/",
    "who": "https://www.who.int/",
    "reuters": "https://www.reuters.com/",
    "ap_news": "https://apnews.com/",
    "imd": "https://mausam.imd.gov.in/",
    "ndma": "https://ndma.gov.in/"
}
```

### Rate Limiting

```python
RATE_LIMIT_REQUESTS = 100  # per user
RATE_LIMIT_PERIOD = 3600   # 1 hour
```

### Confidence Thresholds

```python
CONFIDENCE_THRESHOLD = 0.6
SIMILARITY_THRESHOLD = 0.75
```

---

## 📝 Development

### Add New Agent

1. Create file in `backend/agents/`
2. Inherit from `Agent` class
3. Define role, goal, backstory
4. Add to orchestrator workflow

### Add New Source

1. Edit `source_crawler.py`
2. Add crawling method
3. Add source to config
4. Update RSS feeds list

### Customize Verdicts

Edit `veracity_agent.py` → `_aggregate_verdicts()` method

---

## 🎨 UI Customization

Frontend uses Tailwind CSS. Edit `frontend/src/App.js` to modify:

- Colors: Change `bg-indigo-600` to other Tailwind colors
- Layout: Modify component structure
- Verdict icons: Update `verdictConfig` object

---

## 🚢 Deployment

### Cloud Deployment (Production)

**Backend (Railway/Render):**
1. Push code to GitHub
2. Connect to Railway/Render
3. Add environment variables
4. Deploy

**Frontend (Vercel):**
1. Push to GitHub
2. Import to Vercel
3. Set `REACT_APP_API_URL` to backend URL
4. Deploy

**MongoDB Atlas:**
1. Create cluster
2. Get connection string
3. Update `MONGODB_URI`

**Pinecone:**
- Already cloud-hosted ✅

---

## 🧪 Testing

### Test Single Verification

```bash
curl -X POST http://localhost:5000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "WHO declares new pandemic"}'
```

### Test Crawling

```bash
curl -X POST http://localhost:5000/api/crawl
```

### Check Stats

```bash
curl http://localhost:5000/api/stats
```

---

## 🛠️ Troubleshooting

### "Pinecone index not found"
- Create index in Pinecone dashboard
- Verify `PINECONE_INDEX_NAME` in `.env`

### "MongoDB connection failed"
- Start MongoDB: `mongod`
- Check `MONGODB_URI` in `.env`

### "NLI model loading error"
- Run: `pip install transformers torch`
- Model auto-downloads on first run

### "Rate limit exceeded"
- Adjust `RATE_LIMIT_REQUESTS` in config
- Clear MongoDB interaction history

---

## 📄 License

MIT License - See LICENSE file

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push and create PR

---

## 📞 Support

For issues, create GitHub issue or contact team.

---

**Built with ❤️ for the hackathon**