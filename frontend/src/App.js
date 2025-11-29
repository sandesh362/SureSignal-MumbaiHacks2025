import React, { useState } from 'react';
import { AlertCircle, CheckCircle, XCircle, HelpCircle, TrendingUp, ExternalLink } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('text');
  const [showEli12, setShowEli12] = useState(false);
  const [trending, setTrending] = useState([]);

  const verdictConfig = {
    TRUE: { icon: CheckCircle, color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-200' },
    FALSE: { icon: XCircle, color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-200' },
    MISLEADING: { icon: AlertCircle, color: 'text-yellow-600', bg: 'bg-yellow-50', border: 'border-yellow-200' },
    UNVERIFIED: { icon: HelpCircle, color: 'text-gray-600', bg: 'bg-gray-50', border: 'border-gray-200' }
  };

  const verifyText = async () => {
    if (!inputText.trim()) {
      setError('Please enter a claim to verify');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Verification failed');
      }

      setResult(data.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const verifyUrl = async () => {
    if (!inputUrl.trim()) {
      setError('Please enter a URL to verify');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/verify-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: inputUrl })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Verification failed');
      }

      setResult(data.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchTrending = async () => {
    try {
      const response = await fetch(`${API_URL}/api/trending`);
      const data = await response.json();
      if (data.success) {
        setTrending(data.data);
      }
    } catch (err) {
      console.error('Error fetching trending:', err);
    }
  };

  React.useEffect(() => {
    fetchTrending();
  }, []);

  const VerdictIcon = result ? verdictConfig[result.verdict]?.icon : null;
  const verdictStyle = result ? verdictConfig[result.verdict] : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-indigo-900">VeriPulse</h1>
              <p className="text-sm text-gray-600 mt-1">AI-Powered Misinformation Detection</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={fetchTrending}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
              >
                <TrendingUp size={18} />
                Trending
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-8">
          <div className="flex gap-4 mb-4">
            <button
              onClick={() => setMode('text')}
              className={`px-4 py-2 rounded-lg font-medium transition ${
                mode === 'text'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Verify Text
            </button>
            <button
              onClick={() => setMode('url')}
              className={`px-4 py-2 rounded-lg font-medium transition flex items-center gap-2 ${
                mode === 'url'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <ExternalLink size={18} />
              Verify URL
            </button>
          </div>

          {mode === 'text' ? (
            <div>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste a claim, tweet, or news snippet to verify..."
                className="w-full h-32 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
              />
              <button
                onClick={verifyText}
                disabled={loading}
                className="mt-4 w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 transition"
              >
                {loading ? 'Verifying...' : 'Verify Claim'}
              </button>
            </div>
          ) : (
            <div>
              <input
                type="url"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="https://example.com/article"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
              <button
                onClick={verifyUrl}
                disabled={loading}
                className="mt-4 w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400 transition"
              >
                {loading ? 'Verifying...' : 'Verify URL'}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="text-red-600 flex-shrink-0 mt-0.5" size={20} />
              <p className="text-red-800">{error}</p>
            </div>
          )}
        </div>

        {result && (
          <div className={`bg-white rounded-2xl shadow-lg p-6 mb-8 border-2 ${verdictStyle.border}`}>
            <div className={`flex items-center gap-3 mb-4 p-4 rounded-lg ${verdictStyle.bg}`}>
              {VerdictIcon && <VerdictIcon className={verdictStyle.color} size={32} />}
              <div>
                <h2 className={`text-2xl font-bold ${verdictStyle.color}`}>
                  {result.verdict}
                </h2>
                <p className="text-sm text-gray-600">
                  Confidence: {(result.confidence * 100).toFixed(0)}%
                </p>
              </div>
            </div>

            <div className="mb-4">
              <h3 className="font-semibold text-gray-900 mb-2">Claim:</h3>
              <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">{result.claim}</p>
            </div>

            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-900">Explanation:</h3>
                <button
                  onClick={() => setShowEli12(!showEli12)}
                  className="text-sm text-indigo-600 hover:text-indigo-800"
                >
                  {showEli12 ? 'Show Detailed' : 'Show Simple'}
                </button>
              </div>
              <p className="text-gray-700 leading-relaxed">
                {showEli12
                  ? result.explanation?.eli12_explanation
                  : result.explanation?.detailed_explanation}
              </p>
            </div>

            {result.evidence && result.evidence.length > 0 && (
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Sources:</h3>
                <div className="space-y-3">
                  {result.evidence.map((source, idx) => (
                    <div key={idx} className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <span className="inline-block px-2 py-1 bg-indigo-100 text-indigo-800 text-xs font-semibold rounded uppercase mb-2">
                            {source.source}
                          </span>
                          <h4 className="font-medium text-gray-900 mb-1">{source.title}</h4>
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm text-indigo-600 hover:underline"
                          >
                            {source.url}
                          </a>
                        </div>
                        <span className="text-sm font-medium text-gray-600 ml-4">
                          {(source.relevance_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {trending.length > 0 && (
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <TrendingUp className="text-indigo-600" size={24} />
              Trending Verifications
            </h2>
            <div className="space-y-3">
              {trending.slice(0, 5).map((item, idx) => (
                <div key={idx} className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition">
                  <div className="flex items-start justify-between">
                    <p className="text-gray-800 flex-1">{item.claim}</p>
                    <span className={`ml-4 px-3 py-1 rounded-full text-sm font-semibold ${
                      verdictConfig[item.verdict]?.bg
                    } ${verdictConfig[item.verdict]?.color}`}>
                      {item.verdict}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <p className="text-center text-gray-600 text-sm">
            VeriPulse • Powered by AI • For Demo Purposes Only
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;