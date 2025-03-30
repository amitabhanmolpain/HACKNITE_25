import React, { useState } from 'react';
import './App.css';

function App() {
  const [summary, setSummary] = useState('');
  const [flashcards, setFlashcards] = useState([]);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSummaryUpload = async (event) => {
    setLoading(true);
    setError('');
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload_pdf_summary', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setSummary(data.summary);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFlashcardUpload = async (event) => {
    setLoading(true);
    setError('');
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload_pdf_flashcards', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setFlashcards(data.flashcards);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: chatInput }),
      });
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setChatMessages([...chatMessages, { user: chatInput, ai: data.response }]);
      setChatInput('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="hero-section">
        <h1>AIEdu Platform</h1>
        <p>Empower your learning with AI</p>
      </header>

      {loading && <div className="loading">Loading...</div>}
      {error && <div className="error">{error}</div>}

      <section className="features">
        <div className="feature-card">
          <h2>PDF Summarizer</h2>
          <input type="file" accept=".pdf" onChange={handleSummaryUpload} disabled={loading} />
          {summary && <p className="result">{summary}</p>}
        </div>

        <div className="feature-card">
          <h2>Flashcard Maker</h2>
          <input type="file" accept=".pdf" onChange={handleFlashcardUpload} disabled={loading} />
          {flashcards.length > 0 && (
            <div className="flashcards">
              {flashcards.map((card, index) => (
                <div key={index} className="flashcard">
                  <p><strong>Q:</strong> {card.front}</p>
                  <p><strong>A:</strong> {card.back}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="feature-card">
          <h2>AI Chatbot</h2>
          <div className="chatbox">
            {chatMessages.map((msg, index) => (
              <div key={index} className="chat-message">
                <p className="user-msg"><strong>You:</strong> {msg.user}</p>
                <p className="ai-msg"><strong>AI:</strong> {msg.ai}</p>
              </div>
            ))}
          </div>
          <form onSubmit={handleChatSubmit} className="chat-form">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask me anything..."
              disabled={loading}
            />
            <button type="submit" disabled={loading}>Send</button>
          </form>
        </div>
      </section>
    </div>
  );
}

export default App;