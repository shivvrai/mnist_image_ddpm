import React, { useState, useEffect } from 'react';
import { Database, RefreshCw, LayoutGrid, Play } from 'lucide-react';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

function App() {
  const [digit, setDigit] = useState(7);
  const [guidanceScale, setGuidanceScale] = useState(3.0);
  const [seed, setSeed] = useState(42);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  
  // Load initial archive from localStorage
  const [archive, setArchive] = useState(() => {
    const saved = localStorage.getItem("ddpm_archive");
    return saved ? JSON.parse(saved) : [];
  });
  
  const [progress, setProgress] = useState(0);
  const [modelStatus, setModelStatus] = useState("Checking...");

  // Health check on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(res => res.json())
      .then(data => {
        if (data.status === 'ok') setModelStatus("Online: " + data.loaded_models.join(", "));
      })
      .catch(() => setModelStatus("Offline"));
  }, []);

  // Save archive to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("ddpm_archive", JSON.stringify(archive));
  }, [archive]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (isGenerating) return;
      if (e.key >= '0' && e.key <= '9') {
        setDigit(parseInt(e.key));
      } else if (e.key === 'Enter') {
        handleGenerate();
      } else if (e.key.toLowerCase() === 'r') {
        setSeed(Math.floor(Math.random() * 10000));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [digit, isGenerating]);

  const handleGenerate = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setProgress(0);
    
    // Simulate progress while waiting for the API
    const progressInterval = setInterval(() => {
      setProgress(p => Math.min(p + 5, 95));
    }, 200); // reaches 95% in ~3.8 seconds
    
    try {
      const response = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit,
          guidance_scale: guidanceScale,
          seed,
          model_id: "mnist-ddpm"
        })
      });
      
      const data = await response.json();
      
      clearInterval(progressInterval);
      setProgress(100);
      
      if (data.image_b64) {
        setCurrentImage(data.image_b64);
        setArchive(prev => [{
          id: Date.now(),
          image: data.image_b64,
          digit: data.digit,
          seed: data.seed,
          gs: data.guidance_scale,
          time: data.generation_time_ms
        }, ...prev]);
      }
    } catch (err) {
      console.error(err);
      clearInterval(progressInterval);
      setProgress(0);
    } finally {
      setTimeout(() => setIsGenerating(false), 300); // Give progress bar time to hit 100%
    }
  };

  return (
    <div className="app-container">
      {/* Archive Sidebar */}
      <div className="archive-sidebar">
        <div>
          <h2>Generation Archive</h2>
          <p>Session history</p>
        </div>
        {archive.map((item, index) => (
          <div key={item.id} className="archive-item" onClick={() => {
            setDigit(item.digit);
            setSeed(item.seed);
            setGuidanceScale(item.gs);
            setCurrentImage(item.image);
          }}>
            <img src={item.image} alt="archive" className="archive-thumb" />
            <div className="archive-info">
              <span className="archive-digit">Digit {item.digit}</span>
              <span className="archive-meta">Seed: {item.seed} | GS: {item.gs}</span>
              <span className="archive-meta">{item.time}ms | {index === 0 ? "Latest" : "Cached"}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Main Content */}
      <div className="main-content">
        <header style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>Command Center</h1>
            <p>DDPM Handwriting Generation</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <Database size={16} />
            <span>{modelStatus}</span>
          </div>
        </header>

        {/* Controls */}
        <div className="glass-panel controls-bar">
          <div className="control-group">
            <span className="control-label">Target Digit</span>
            <div className="digits-container">
              {[0,1,2,3,4,5,6,7,8,9].map(num => (
                <button 
                  key={num}
                  className={`digit-btn ${digit === num ? 'active' : ''}`}
                  onClick={() => setDigit(num)}
                  disabled={isGenerating}
                >
                  {num}
                </button>
              ))}
            </div>
          </div>

          <div className="control-group" style={{ marginLeft: '1rem' }}>
            <span className="control-label">Guidance Scale ({guidanceScale.toFixed(1)})</span>
            <input 
              type="range" 
              min="0" max="10" step="0.1"
              value={guidanceScale}
              onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
              disabled={isGenerating}
            />
          </div>
          
          <div className="control-group" style={{ marginLeft: '1rem' }}>
            <span className="control-label">Seed</span>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <span style={{ fontFamily: 'monospace', fontSize: '1.1rem' }}>{seed}</span>
              <button 
                className="digit-btn" 
                style={{ width: '30px', height: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                onClick={() => setSeed(Math.floor(Math.random() * 10000))}
                title="Randomize Seed (R)"
                disabled={isGenerating}
              >
                <RefreshCw size={14} />
              </button>
            </div>
          </div>

          <button 
            className="generate-btn" 
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            {isGenerating ? <RefreshCw size={18} className="spin" /> : <Play size={18} fill="currentColor" />}
            {isGenerating ? 'Generating...' : 'Generate (Enter)'}
          </button>
        </div>

        {isGenerating && (
          <div style={{ padding: '1rem', marginBottom: '2rem', backgroundColor: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.5)', borderRadius: '8px', color: '#fef08a', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <RefreshCw size={16} className="spin" />
            <span>Generating on CPU (Native Windows GPU unsupported). This requires 600 model passes and takes 1-3 minutes. Please wait...</span>
          </div>
        )}

        {/* Viewer */}
        <div className="viewer-area">
          <div className="preview-box">
            <div className="status-badge">
              <div className={`status-dot ${isGenerating ? 'generating' : ''}`}></div>
              {isGenerating ? 'Diffusion Active' : 'Idle'}
            </div>
            
            {/* The noise layer shown when generating */}
            <div className={`noise-layer ${isGenerating ? 'active' : ''}`}></div>
            
            {/* The final image */}
            {currentImage && (
              <img 
                src={currentImage} 
                alt="Generated" 
                className={`preview-image ${currentImage && !isGenerating ? 'loaded' : ''}`}
                style={{ opacity: isGenerating ? 0 : 1 }}
              />
            )}
            
            {/* Progress Bar */}
            {isGenerating && (
              <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            )}
          </div>
          
          {/* Comparison Mode Slot */}
          <div className="preview-box" style={{ opacity: 0.5, borderStyle: 'dashed' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', color: 'var(--text-secondary)' }}>
              <LayoutGrid size={32} />
              <p>Comparison Slot</p>
              <p style={{fontSize: '0.75rem', textAlign: 'center', padding: '0 2rem'}}>Click an archive item to load into preview for comparison.</p>
            </div>
          </div>
        </div>
        
      </div>
    </div>
  );
}

export default App;
