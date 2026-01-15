import React, { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const Landing = () => {
  const [driveUrl, setDriveUrl] = useState('');
  const [videoName, setVideoName] = useState('');
  const [jobId, setJobId] = useState('');
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [polling, setPolling] = useState(false);

  const canSubmit = useMemo(() => driveUrl.trim().length > 0, [driveUrl]);

  useEffect(() => {
    let interval;
    if (jobId && polling) {
      interval = setInterval(() => pollStatus(jobId), 5000);
      pollStatus(jobId); // immediate poll
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId, polling]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    setStatus('pending');
    setProgress(0);
    setJobId('');

    try {
      const resp = await fetch(`${API_BASE}/api/videos/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drive_video_url: driveUrl.trim(),
          video_name: videoName.trim() || 'Untitled Video',
        }),
      });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || 'Failed to start processing');
      }

      const data = await resp.json();
      setJobId(data.job_id);
      setStatus(data.status || 'pending');
      setProgress(data.progress || 0);
      setPolling(true);
    } catch (err) {
      setError(err.message || 'Something went wrong');
      setStatus('failed');
    }
  };

  const pollStatus = async (id) => {
    try {
      const resp = await fetch(`${API_BASE}/api/videos/status/${id}`);
      if (!resp.ok) return;
      const data = await resp.json();
      setStatus(data.status);
      setProgress(data.progress || 0);

      if (data.status === 'completed') {
        setPolling(false);
        await fetchResults(id);
      }

      if (data.status === 'failed') {
        setPolling(false);
        setError('Job failed. Check backend logs for details.');
      }
    } catch (err) {
      console.warn('Status poll failed', err);
    }
  };

  const fetchResults = async (id) => {
    try {
      const resp = await fetch(`${API_BASE}/api/videos/results/${id}`);
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || 'Failed to fetch results');
      }
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch results');
    }
  };

  const renderTopics = () => {
    if (!result?.topics?.length) return null;
    return result.topics.map((topic, idx) => (
      <div className="card" key={`${topic.title}-${idx}`}>
        <div className="card-header">
          <div className="pill">Topic {idx + 1}</div>
          <div className="timestamp">{topic.timestamp_range?.join(' — ')}</div>
        </div>
        <h3>{topic.title}</h3>
        {topic.summary && <p className="muted">{topic.summary}</p>}
        {topic.key_points?.length > 0 && (
          <ul className="bullets">
            {topic.key_points.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        )}
        {topic.frames?.length > 0 && (
          <div className="frames-grid">
            {topic.frames.slice(0, 4).map((f, i) => (
              <a
                className="frame-thumb"
                href={f.drive_url}
                target="_blank"
                rel="noreferrer"
                key={i}
              >
                <div className="frame-meta">
                  <span>{f.timestamp}</span>
                  <span className="pill pill-ghost">{f.type || 'frame'}</span>
                </div>
                <div className="frame-desc">{f.description || 'Frame'}</div>
              </a>
            ))}
          </div>
        )}
      </div>
    ));
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Gemini + Google Drive</p>
          <h1>Long-Form Video Intelligence</h1>
          <p className="muted">
            Paste a Google Drive video link, kick off processing, and see transcript + key frames + insights.
          </p>
        </div>
        <div className="status-chip">
          <span className={`dot dot-${status === 'completed' ? 'green' : status === 'failed' ? 'red' : 'amber'}`} />
          <span>{status === 'idle' ? 'Idle' : status}</span>
        </div>
      </header>

      <form className="card form" onSubmit={handleSubmit}>
        <div className="field">
          <label>Google Drive Video URL</label>
          <input
            type="url"
            placeholder="https://drive.google.com/file/d/FILE_ID/view"
            value={driveUrl}
            onChange={(e) => setDriveUrl(e.target.value)}
            required
          />
        </div>

        <div className="field">
          <label>Video Name (optional)</label>
          <input
            type="text"
            placeholder="My Seminar"
            value={videoName}
            onChange={(e) => setVideoName(e.target.value)}
          />
        </div>

        <div className="actions">
          <button type="submit" disabled={!canSubmit || status === 'pending' || polling}>
            {status === 'pending' || polling ? 'Processing…' : 'Start Processing'}
          </button>
          {jobId && <span className="muted">Job ID: {jobId}</span>}
        </div>

        {error && <div className="error">{error}</div>}

        {(status !== 'idle' && status !== 'failed') && (
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
            <div className="progress-label">{Math.round(progress * 100)}%</div>
          </div>
        )}
      </form>

      {result && (
        <section className="grid">
          <div className="card">
            <div className="card-header">
              <div className="pill">Summary</div>
              <div className="timestamp">Duration: {result.duration ? `${Math.round(result.duration / 60)} min` : '—'}</div>
            </div>
            <h3>Executive Summary</h3>
            <p className="muted">{result.executive_summary || 'No summary yet.'}</p>
            {result.key_takeaways?.length > 0 && (
              <div>
                <h4>Key Takeaways</h4>
                <ul className="bullets">
                  {result.key_takeaways.map((k, i) => (
                    <li key={i}>{k}</li>
                  ))}
                </ul>
              </div>
            )}
            {result.entities && (
              <div className="chips">
                {Object.entries(result.entities).flatMap(([type, items]) =>
                  (items || []).map((item, idx) => (
                    <span className="chip" key={`${type}-${idx}`}>
                      {item}
                    </span>
                  ))
                )}
              </div>
            )}
          </div>

          <div className="stack">
            {renderTopics()}
          </div>
        </section>
      )}
    </div>
  );
};

export default Landing;
