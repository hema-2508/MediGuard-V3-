import { useMemo, useState } from 'react';

const API_URL = '/api/v1/orchestrate';

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!inputText.trim()) {
      setError('Please enter a prescription or medication description.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_text: inputText.trim() }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.detail?.[0]?.msg || payload?.detail || 'Request failed.');
      }

      setResult(payload);
    } catch (err) {
      setError(err.message || 'Unexpected error while contacting the API.');
    } finally {
      setLoading(false);
    }
  };

  const summary = useMemo(() => {
    if (!result?.summary) return null;
    return result.summary;
  }, [result]);

  return (
    <div className="app-shell">
      <header className="hero-card">
        <div>
          <p className="eyebrow">MediGuard</p>
          <h1>Medication intelligence at a glance</h1>
          <p className="hero-copy">
            Paste a prescription, review extracted medicines, and inspect interaction, safety, and explainability insights.
          </p>
        </div>
        <form onSubmit={handleSubmit} className="input-card">
          <label htmlFor="drug-input">Prescription or medication text</label>
          <textarea
            id="drug-input"
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            placeholder="Example: Take amoxicillin and ibuprofen for pain"
            rows={5}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </form>
      </header>

      {error ? <div className="status-card error">{error}</div> : null}

      {loading ? <div className="status-card">Analyzing your prescription…</div> : null}

      {result ? (
        <main className="results-grid">
          <section className="panel">
            <h2>Summary</h2>
            <div className="summary-grid">
              <div>
                <span className="metric-label">Medicines</span>
                <strong>{summary?.medicine_count ?? 0}</strong>
              </div>
              <div>
                <span className="metric-label">Risks</span>
                <strong>{summary?.risk_count ?? 0}</strong>
              </div>
            </div>
            <p className="chat-response">{result.chat_response}</p>
          </section>

          <section className="panel">
            <h2>Extracted medicines</h2>
            <div className="stack">
              {result.medicines?.length ? (
                result.medicines.map((medicine, index) => (
                  <article key={`${medicine.name}-${index}`} className="card">
                    <h3>{medicine.name}</h3>
                    <p><strong>Strength:</strong> {medicine.strength || 'Not provided'}</p>
                    <p><strong>Confidence:</strong> {medicine.confidence?.toFixed(2) ?? '0.00'}</p>
                  </article>
                ))
              ) : (
                <p>No medicines detected.</p>
              )}
            </div>
          </section>

          <section className="panel">
            <h2>Interaction prediction</h2>
            <div className="stack">
              {result.medicines?.map((medicine, index) => (
                <article key={`risk-${index}`} className="card">
                  <h3>{medicine.name}</h3>
                  <p>{medicine.risk_summary?.common_adverse_reactions?.length ? medicine.risk_summary.common_adverse_reactions.join(', ') : 'No obvious adverse reactions listed.'}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Risk assessment</h2>
            <div className="stack">
              {result.medicines?.map((medicine, index) => (
                <article key={`risk-assessment-${index}`} className="card">
                  <h3>{medicine.name}</h3>
                  <p>{medicine.risk_summary?.common_adverse_reactions?.length ? medicine.risk_summary.common_adverse_reactions.join(', ') : 'No acute risk signals.'}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Symptom reasoning</h2>
            <div className="stack">
              {result.medicines?.map((medicine, index) => (
                <article key={`symptom-${index}`} className="card">
                  <h3>{medicine.name}</h3>
                  <p>{medicine.symptom_reasoning?.known_adverse_effect ? 'Known adverse effect match detected.' : 'No direct symptom match identified.'}</p>
                  <p><strong>Severity:</strong> {medicine.symptom_reasoning?.severity || 'unknown'}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Explainability</h2>
            <div className="stack">
              {result.medicines?.map((medicine, index) => (
                <article key={`explain-${index}`} className="card">
                  <h3>{medicine.name}</h3>
                  <p>{medicine.explanation?.explanation || 'No explanation available.'}</p>
                </article>
              ))}
            </div>
          </section>
        </main>
      ) : null}
    </div>
  );
}

export default App;
