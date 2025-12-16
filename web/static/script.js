// Hate-Speech Detection Web Interface - JavaScript

// API base URL
const API_BASE = '';

// Elements
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContent = document.getElementById('resultsContent');
const statusBadge = document.getElementById('statusBadge');
const examplesGrid = document.getElementById('examplesGrid');

// Check system status on load
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        updateStatusBadge(data);
    } catch (error) {
        updateStatusBadge({ status: 'offline' });
    }
}

// Update status badge
function updateStatusBadge(data) {
    const dot = statusBadge.querySelector('.status-dot');
    const text = statusBadge.querySelector('.status-text');
    
    if (data.status === 'online') {
        dot.style.background = 'var(--success)';
        if (data.model_trained) {
            text.textContent = 'Model Ready (Trained)';
        } else {
            text.textContent = 'Model Ready (Untrained)';
            dot.style.background = 'var(--warning)';
        }
    } else {
        dot.style.background = 'var(--danger)';
        text.textContent = 'Offline';
    }
}

// Load examples
async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE}/api/examples`);
        const data = await response.json();
        
        examplesGrid.innerHTML = '';
        data.examples.forEach(category => {
            category.texts.forEach(text => {
                const btn = document.createElement('button');
                btn.className = 'example-btn';
                btn.textContent = text;
                btn.onclick = () => {
                    textInput.value = text;
                    textInput.focus();
                };
                examplesGrid.appendChild(btn);
            });
        });
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

// Analyze text
async function analyzeText() {
    const text = textInput.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    // Show loading state
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/api/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred');
        }
    } catch (error) {
        showError('Failed to connect to the server. Make sure the server is running.');
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    const { prediction, confidence, probabilities, model_status } = data;
    
   const html = `
        <div class="result-card">
            <div class="result-prediction">
                <div class="prediction-label">Prediction</div>
                <div class="prediction-value ${prediction}">
                    ${formatPrediction(prediction)}
                </div>
                <div class="confidence-badge">
                    ${(confidence * 100).toFixed(1)}% confidence
                </div>
                ${model_status === 'untrained' ? 
                    '<p style="margin-top: 1rem; color: var(--warning); font-size: 0.9rem;">⚠️ Model is untrained - predictions are random. Train the model for accurate results.</p>' : 
                    ''
                }
            </div>
            
            <div class="probabilities-section">
                <h3 style="margin-bottom: 1rem; color: var(--text-secondary); font-size: 1rem;">Class Probabilities</h3>
                
                <div class="probability-bar">
                    <div class="probability-label">
                        <span>😊 Neutral</span>
                        <span>${(probabilities.neutral * 100).toFixed(1)}%</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill neutral" style="width: ${probabilities.neutral * 100}%"></div>
                    </div>
                </div>
                
                <div class="probability-bar">
                    <div class="probability-label">
                        <span>⚠️ Offensive</span>
                        <span>${(probabilities.offensive * 100).toFixed(1)}%</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill offensive" style="width: ${probabilities.offensive * 100}%"></div>
                    </div>
                </div>
                
                <div class="probability-bar">
                    <div class="probability-label">
                        <span>🚫 Hate Speech</span>
                        <span>${(probabilities.hate_speech * 100).toFixed(1)}%</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill hate_speech" style="width: ${probabilities.hate_speech * 100}%"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultsContent.innerHTML = html;
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Format prediction label
function formatPrediction(prediction) {
    const labels = {
        'neutral': '😊 Neutral',
        'offensive': '⚠️ Offensive',
        'hate_speech': '🚫 Hate Speech'
    };
    return labels[prediction] || prediction;
}

// Show error
function showError(message) {
    resultsContent.innerHTML = `
        <div class="empty-state" style="color: var(--danger);">
            <div class="empty-icon">⚠️</div>
            <p>${message}</p>
        </div>
    `;
}

// Clear input
function clearInput() {
    textInput.value = '';
    resultsContent.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <p>Enter text and click "Analyze" to see results</p>
        </div>
    `;
    textInput.focus();
}

// Event listeners
analyzeBtn.addEventListener('click', analyzeText);
clearBtn.addEventListener('click', clearInput);

// Allow Enter key to submit (Ctrl+Enter for new line)
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
        e.preventDefault();
        analyzeText();
    }
});

// Initialize
checkStatus();
loadExamples();
