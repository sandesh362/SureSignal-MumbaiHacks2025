# VeriPulse - 10-Minute Quickstart Guide

## ⚡ Super Fast Setup

### Prerequisites Checklist
- [ ] Python 3.9+ installed
- [ ] Node.js 16+ installed  
- [ ] OpenAI API key (get from https://platform.openai.com/api-keys)
- [ ] Google Gemini API key (get from https://makersuite.google.com/app/apikey)
- [ ] Pinecone account (sign up at https://www.pinecone.io/)

---

## 🚀 Option 1: Docker (Easiest - 5 minutes)

```bash
# 1. Clone repo
git clone <your-repo>
cd veripulse

# 2. Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-gemini-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=veripulse-evidence
MONGODB_URI=mongodb://mongodb:27017/
EOF

# 3. Start everything
docker-compose up -d

# 4. Wait 30 seconds, then visit:
# Frontend: http://localhost:3000
# Backend: http://localhost:5000
```

**That's it!** Skip to "Testing" section below.

---

## 🛠️ Option 2: Manual Setup (10 minutes)

### Step 1: Setup Pinecone (2 mins)

1. Go to https://www.pinecone.io/ and sign in
2. Click "Create Index"
   - Name: `veripulse-evidence`
   - Dimensions: `768`
   - Metric: `cosine`
3. Copy your API key

### Step 2: Create Project (1 min)

```bash
# Clone or create directory
mkdir veripulse
cd veripulse

# Create structure
mkdir -p backend/agents backend/services frontend/src
```

### Step 3: Environment Setup (1 min)

```bash
# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=your-openai-key-here
GOOGLE_API_KEY=your-gemini-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=veripulse-evidence
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=veripulse
EOF

# Edit with your actual keys
nano .env
```

### Step 4: Backend Setup (3 mins)

```bash
# Copy all backend files from artifacts to backend/
# (Copy: config.py, app.py, agents/*, services/*)

cd backend

# Install dependencies
pip install flask flask-cors python-dotenv crewai langchain \
  langchain-openai langchain-google-genai openai google-generativeai \
  sentence-transformers transformers torch pinecone-client pymongo \
  tweepy praw requests beautifulsoup4 feedparser newspaper3k \
  pandas numpy pydantic scikit-learn

# Start MongoDB (in new terminal)
mongod --dbpath ~/data/db

# Run backend
python app.py
```

Backend should start at http://localhost:5000

### Step 5: Frontend Setup (3 mins)

```bash
# In new terminal
cd frontend

# Create package.json
cat > package.json << 'EOF'
{
  "name": "veripulse-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "lucide-react": "^0.263.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  }
}
EOF

# Install
npm install

# Create public/index.html
mkdir -p public
cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>VeriPulse</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
EOF

# Setup Tailwind
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init

# Create tailwind.config.js
cat > tailwind.config.js << 'EOF'
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: { extend: {} },
  plugins: [],
}
EOF

# Create src/index.css
mkdir -p src
cat > src/index.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF

# Create src/index.js
cat > src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
EOF

# Copy App.js from artifact to src/

# Start frontend
npm start
```

Frontend should open at http://localhost:3000

---

## 🧪 Testing Your Setup

### Test 1: Backend Health Check

```bash
curl http://localhost:5000/health
# Expected: {"status": "healthy", ...}
```

### Test 2: Simple Verification

```bash
curl -X POST http://localhost:5000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Tsunami warning issued in Mumbai"}'
```

Expected response (after ~8 seconds):
```json
{
  "success": true,
  "data": {
    "claim": "Tsunami warning issued in Mumbai",
    "verdict": "FALSE",
    "confidence": 0.85,
    "explanation": {...},
    "evidence": [...]
  }
}
```

### Test 3: Frontend

1. Open http://localhost:3000
2. Enter: "WHO declares new pandemic"
3. Click "Verify Claim"
4. Should see verdict in 5-10 seconds

---

## 🐛 Troubleshooting

### "Module not found" errors

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### "Pinecone index not found"

```python
# Create index manually
import pinecone
pinecone.init(api_key="your-key", environment="gcp-starter")
pinecone.create_index("veripulse-evidence", dimension=768, metric="cosine")
```

### "MongoDB connection failed"

```bash
# Start MongoDB
mongod --dbpath ~/data/db

# Or install MongoDB:
# Mac: brew install mongodb-community
# Linux: sudo apt install mongodb
# Windows: Download from mongodb.com
```

### "Cannot connect to backend"

Check `.env` has correct API keys:
```bash
cat .env | grep API_KEY
```

### Port already in use

```bash
# Change backend port in app.py:
app.run(port=5001)  # Instead of 5000

# Update frontend API URL:
# In App.js: const API_URL = 'http://localhost:5001'
```

---

## 📦 Quick Deploy to Cloud

### Backend → Railway

```bash
# Push to GitHub first
git add .
git commit -m "Initial commit"
git push

# Then in Railway.app:
1. New Project → Deploy from GitHub
2. Select repo
3. Add all environment variables from .env
4. Deploy!
```

### Frontend → Vercel

```bash
cd frontend
vercel

# Or via Vercel dashboard:
# Import → Select repo → Set root to "frontend"
```

---

## 🎯 Next Steps

### Add Sample Data

```bash
# In backend directory
python -c "
from agents.orchestrator_agent import OrchestratorAgent
orch = OrchestratorAgent()
orch.crawl_and_index_sources()
print('✅ Indexed sources')
"
```

### Enable Bot (Optional)

1. Get Twitter/Reddit API keys
2. Add to `.env`:
   ```
   TWITTER_BEARER_TOKEN=your-token
   REDDIT_CLIENT_ID=your-id
   REDDIT_CLIENT_SECRET=your-secret
   ```
3. Run bot monitor:
   ```bash
   python backend/bot_monitor.py
   ```

### Customize Sources

Edit `backend/config.py`:
```python
TRUSTED_SOURCES = {
    "your_source": "https://example.com",
    ...
}
```

---

## 📚 Documentation

- Full README: [README.md](README.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Deployment: [DEPLOYMENT.md](DEPLOYMENT.md)
- Demo Script: [DEMO_SCRIPT.md](DEMO_SCRIPT.md)

---

## 💬 Need Help?

- Check logs: `docker-compose logs -f` (Docker) or `python app.py` output
- Verify environment variables: `cat .env`
- Test API directly: Use Postman or curl
- Common issues documented in [DEPLOYMENT.md](DEPLOYMENT.md)

---

## ✅ Success Checklist

- [ ] Backend running on http://localhost:5000
- [ ] Frontend running on http://localhost:3000
- [ ] MongoDB connected
- [ ] Pinecone index created
- [ ] Test verification works
- [ ] Can see verdict and sources
- [ ] ELI12 toggle works

---

**If all boxes checked → You're ready to demo! 🎉**

Total setup time: 5-10 minutes with Docker, 10-15 minutes manual.

Questions? Review the full [README.md](README.md) for detailed explanations!