import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Database, RefreshCw, LayoutGrid, Play, Zap, PenTool, Trash2, Eye } from 'lucide-react';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
const CANVAS_SIZE = 280; // 10x MNIST resolution for comfortable drawing

function App() {
  const [digit, setDigit] = useState(7);
  const [guidanceScale, setGuidanceScale] = useState(3.0);
  const [seed, setSeed] = useState(42);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  const [mode, setMode] = useState('noise'); // 'noise' or 'sketch'
  const [strength, setStrength] = useState(0.5);
  const [prediction, setPrediction] = useState(null); // { digit, confidence }
  const [pendingVerification, setPendingVerification] = useState(false);
  
  // Canvas drawing state
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  
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

  // Initialize canvas with black background when mode switches to sketch
  useEffect(() => {
    if (mode === 'sketch' && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }
    setPrediction(null);
    setPendingVerification(false);
  }, [mode]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (isGenerating) return;
      // Don't trigger shortcuts when typing in an input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      if (e.key >= '0' && e.key <= '9') {
        setDigit(parseInt(e.key));
      } else if (e.key === 'Enter') {
        if (mode === 'noise') handleGenerate();
        else if (pendingVerification) handleConfirmGeneration();
        else handleGenerateFromSketch();
      } else if (e.key.toLowerCase() === 'r') {
        setSeed(Math.floor(Math.random() * 10000));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [digit, isGenerating, mode, pendingVerification]);

  // ── Canvas drawing handlers ──
  const getCanvasCoords = useCallback((e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Support both mouse and touch events
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }, []);

  const startDrawing = useCallback((e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { x, y } = getCanvasCoords(e);
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 28;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    setIsDrawing(true);
    setPrediction(null);
    setPendingVerification(false);
  }, [getCanvasCoords]);

  const draw = useCallback((e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { x, y } = getCanvasCoords(e);
    
    ctx.lineTo(x, y);
    ctx.stroke();
  }, [isDrawing, getCanvasCoords]);

  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const clearCanvas = useCallback(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setPrediction(null);
    setPendingVerification(false);
  }, []);

  // ── Standard generation (from noise) ──
  const handleGenerate = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setProgress(0);
    
    const progressInterval = setInterval(() => {
      setProgress(p => Math.min(p + 5, 95));
    }, 200);
    
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
          time: data.generation_time_ms,
          type: 'noise'
        }, ...prev]);
      }
    } catch (err) {
      console.error(err);
      clearInterval(progressInterval);
      setProgress(0);
    } finally {
      setTimeout(() => setIsGenerating(false), 300);
    }
  };

  // ── Sketch-to-digit Step 1: Classify ──
  const handleGenerateFromSketch = async () => {
    if (isGenerating || pendingVerification || !canvasRef.current) return;
    setIsGenerating(true);
    setPrediction(null);

    const sketchB64 = canvasRef.current.toDataURL('image/png');

    try {
      const response = await fetch(`${API_URL}/classify-sketch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sketch_b64: sketchB64 })
      });
      const data = await response.json();
      
      if (data.predicted_digit != null) {
        setDigit(data.predicted_digit);
        setPrediction({
          digit: data.predicted_digit,
          confidence: data.confidence
        });
      }
      setPendingVerification(true);
    } catch (err) {
      console.error("Classification failed:", err);
    } finally {
      setIsGenerating(false);
    }
  };

  // ── Sketch-to-digit Step 2: Diffuse ──
  const handleConfirmGeneration = async () => {
    if (isGenerating || !canvasRef.current) return;
    setIsGenerating(true);
    setPendingVerification(false);
    setProgress(0);

    const sketchB64 = canvasRef.current.toDataURL('image/png');

    // Estimate progress: fewer steps when strength < 1
    const estimatedSteps = Math.ceil(strength * 300);
    const stepDuration = estimatedSteps > 0 ? (3800 * strength) / 95 : 200;
    
    const progressInterval = setInterval(() => {
      setProgress(p => Math.min(p + 5, 95));
    }, Math.max(stepDuration, 100));

    try {
      const response = await fetch(`${API_URL}/generate-from-sketch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          digit, // using the user-verified digit
          guidance_scale: guidanceScale,
          seed,
          model_id: "mnist-ddpm",
          sketch_b64: sketchB64,
          strength
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
          time: data.generation_time_ms,
          type: 'sketch',
          strength,
          predictedDigit: prediction?.digit,
          confidence: prediction?.confidence
        }, ...prev]);
      }
    } catch (err) {
      console.error(err);
      clearInterval(progressInterval);
      setProgress(0);
    } finally {
      setTimeout(() => setIsGenerating(false), 300);
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
              <span className="archive-digit">
                {`Digit ${item.digit}`}
                {item.type === 'sketch' && <span style={{ fontSize: '0.7rem', marginLeft: '0.5rem', opacity: 0.7 }}>✏️ Sketch</span>}
                {item.predictedDigit != null && <span style={{ fontSize: '0.65rem', marginLeft: '0.3rem', opacity: 0.6 }}>🤖 {(item.confidence * 100).toFixed(0)}%</span>}
              </span>
              <span className="archive-meta">Seed: {item.seed} | GS: {item.gs}</span>
              <span className="archive-meta">
                {item.time}ms
                {item.type === 'sketch' && ` | Str: ${item.strength}`}
                {index === 0 ? " | Latest" : " | Cached"}
              </span>
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {/* Mode Toggle */}
            <div className="mode-toggle">
              <button
                className={mode === 'noise' ? 'active' : ''}
                onClick={() => setMode('noise')}
              >
                <Zap size={14} /> From Noise
              </button>
              <button
                className={mode === 'sketch' ? 'active' : ''}
                onClick={() => setMode('sketch')}
              >
                <PenTool size={14} /> Sketch-to-Digit
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <Database size={16} />
              <span>{modelStatus}</span>
            </div>
          </div>
        </header>

        {/* Prediction Result Banner (sketch mode only) - Shown during pendingVerification */}
        {prediction && mode === 'sketch' && pendingVerification && (
          <div style={{
            padding: '1rem 1.5rem',
            marginBottom: '1.5rem',
            background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.15))',
            border: '2px solid rgba(34, 197, 94, 0.6)',
            borderRadius: '12px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '1rem',
            animation: 'fadeIn 0.4s ease-out'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <Eye size={24} style={{ color: '#22c55e', flexShrink: 0 }} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                <span style={{ color: '#bbf7d0', fontSize: '1.1rem', fontWeight: 600 }}>
                  Classifier detected: <span style={{ fontSize: '1.5rem', fontWeight: 700, color: '#4ade80' }}>{prediction.digit}</span>
                </span>
                <span style={{ color: 'rgba(187, 247, 208, 0.8)', fontSize: '0.85rem' }}>
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%. Is this correct? You can change it below.
                </span>
              </div>
            </div>
            
            {/* Dedicated verify buttons here */}
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button 
                className="generate-btn" 
                onClick={handleConfirmGeneration}
                disabled={isGenerating}
                style={{ backgroundColor: '#16a34a', border: 'none', padding: '0.6rem 1.2rem', boxShadow: '0 4px 14px rgba(22, 163, 74, 0.4)' }}
              >
                <Play size={16} fill="currentColor" /> Confirm & Diffuse
              </button>
              <button
                className="generate-btn"
                onClick={() => setPendingVerification(false)}
                disabled={isGenerating}
                style={{ backgroundColor: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid rgba(255,255,255,0.2)' }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="glass-panel controls-bar" style={{ opacity: pendingVerification ? 1 : 1, transition: 'opacity 0.3s' }}>
          {/* Target digit shown for noise OR if pendingVerification is active so user can correct it */}
          {(mode === 'noise' || pendingVerification) && (
            <div className="control-group">
              <span className="control-label">Target Digit</span>
              <div className="digits-container">
                {[0,1,2,3,4,5,6,7,8,9].map(num => (
                  <button 
                    key={num}
                    className={`digit-btn ${digit === num ? 'active' : ''}`}
                    onClick={() => setDigit(num)}
                    disabled={isGenerating && !pendingVerification}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>
          )}

          {mode === 'noise' && (
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
          )}
          
          <div className="control-group" style={{ marginLeft: mode === 'noise' || pendingVerification ? '1rem' : '0' }}>
            <span className="control-label">Seed</span>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <span style={{ fontFamily: 'monospace', fontSize: '1.1rem' }}>{seed}</span>
              <button 
                className="digit-btn" 
                style={{ width: '30px', height: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                onClick={() => setSeed(Math.floor(Math.random() * 10000))}
                title="Randomize Seed (R)"
                disabled={isGenerating && !pendingVerification}
              >
                <RefreshCw size={14} />
              </button>
            </div>
          </div>

          {/* Strength slider - only in sketch mode */}
          {mode === 'sketch' && (
            <div className="control-group" style={{ marginLeft: '1rem' }}>
              <span className="control-label">
                Denoising Strength <span className="strength-value">{strength.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min="0" max="1" step="0.05"
                value={strength}
                onChange={(e) => setStrength(parseFloat(e.target.value))}
                disabled={isGenerating && !pendingVerification}
              />
            </div>
          )}

          {!pendingVerification && (
            <button 
              className="generate-btn" 
              onClick={mode === 'noise' ? handleGenerate : handleGenerateFromSketch}
              disabled={isGenerating}
            >
              {isGenerating ? <RefreshCw size={18} className="spin" /> : (mode === 'sketch' ? <Eye size={18} /> : <Play size={18} fill="currentColor" />)}
              {isGenerating ? 'Processing...' : (mode === 'sketch' ? 'Predict & Generate' : 'Generate (Enter)')}
            </button>
          )}
        </div>

        {isGenerating && !pendingVerification && (
          <div style={{ padding: '1rem', marginBottom: '2rem', backgroundColor: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.5)', borderRadius: '8px', color: '#fef08a', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <RefreshCw size={16} className="spin" />
            <span>
              {mode === 'sketch'
                ? `Running partial diffusion (${Math.ceil(strength * 300)} of 300 steps). This takes ${Math.max(1, Math.ceil(strength * 3))} minute(s). Please wait...`
                : 'Generating on CPU (Native Windows GPU unsupported). This requires 600 model passes and takes 1-3 minutes. Please wait...'
              }
            </span>
          </div>
        )}

        {/* Viewer */}
        <div className="viewer-area">
          {/* Preview Box (generated output) */}
          <div className="preview-box">
            <div className="status-badge">
              <div className={`status-dot ${isGenerating && !pendingVerification ? 'generating' : ''}`}></div>
              {isGenerating && !pendingVerification ? 'Diffusion Active' : (pendingVerification ? 'Waiting for Confirm' : 'Idle')}
            </div>
            
            {/* The noise layer shown when generating */}
            <div className={`noise-layer ${isGenerating && !pendingVerification ? 'active' : ''}`}></div>
            
            {/* The final image */}
            {currentImage && (
              <img 
                src={currentImage} 
                alt="Generated" 
                className={`preview-image ${currentImage && !isGenerating ? 'loaded' : ''}`}
                style={{ opacity: isGenerating && !pendingVerification ? 0 : 1 }}
              />
            )}
            
            {/* Progress Bar */}
            {isGenerating && !pendingVerification && (
              <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            )}
          </div>
          
          {/* Right side: Canvas (sketch mode) or Comparison Slot (noise mode) */}
          {mode === 'sketch' ? (
            <div className="canvas-container">
              <span className="canvas-label">Draw Here</span>
              <div className="canvas-grid"></div>
              <canvas
                ref={canvasRef}
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                style={{ width: '100%', height: '100%' }}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
              <div className="canvas-actions">
                <button className="canvas-action-btn" onClick={clearCanvas} disabled={isGenerating && !pendingVerification}>
                  <Trash2 size={12} /> Clear
                </button>
              </div>
            </div>
          ) : (
            <div className="preview-box" style={{ opacity: 0.5, borderStyle: 'dashed' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', color: 'var(--text-secondary)' }}>
                <LayoutGrid size={32} />
                <p>Comparison Slot</p>
                <p style={{fontSize: '0.75rem', textAlign: 'center', padding: '0 2rem'}}>Click an archive item to load into preview for comparison.</p>
              </div>
            </div>
          )}
        </div>
        
      </div>
    </div>
  );
}

export default App;
