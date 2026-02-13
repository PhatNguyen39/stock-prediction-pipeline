// ===== State =====
let trainingPollInterval = null;

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    refreshStatus();
});

// ===== Error Handling =====
function showError(msg) {
    const banner = document.getElementById('error-banner');
    document.getElementById('error-message').textContent = msg;
    banner.classList.remove('hidden');
    setTimeout(dismissError, 8000);
}

function dismissError() {
    document.getElementById('error-banner').classList.add('hidden');
}

// ===== Model Status =====
async function refreshStatus() {
    try {
        const res = await fetch('/');
        const data = await res.json();

        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        const trained = document.getElementById('last-trained');

        if (data.model_loaded) {
            dot.className = 'status-dot green';
            text.textContent = 'Model loaded';
        } else {
            dot.className = 'status-dot red';
            text.textContent = 'Not loaded';
        }

        trained.textContent = data.last_trained
            ? new Date(data.last_trained).toLocaleString()
            : 'Never';
    } catch (e) {
        // Silently fail on status check
    }
}

// ===== Training =====
async function startTraining() {
    const btn = document.getElementById('train-btn');
    const btnText = document.getElementById('train-btn-text');
    const spinner = document.getElementById('train-spinner');
    const statusMsg = document.getElementById('training-status-msg');

    btn.disabled = true;
    btnText.textContent = 'Training...';
    spinner.classList.remove('hidden');
    statusMsg.classList.remove('hidden');
    statusMsg.textContent = 'Training started. This may take a few minutes...';

    try {
        const res = await fetch('/train', { method: 'POST' });
        const data = await res.json();

        if (data.status === 'skipped') {
            statusMsg.textContent = data.message;
            btn.disabled = false;
            btnText.textContent = 'Train Model';
            spinner.classList.add('hidden');
            return;
        }

        // Poll for completion
        trainingPollInterval = setInterval(pollTrainingStatus, 3000);
    } catch (e) {
        showError('Failed to start training: ' + e.message);
        btn.disabled = false;
        btnText.textContent = 'Train Model';
        spinner.classList.add('hidden');
        statusMsg.classList.add('hidden');
    }
}

async function pollTrainingStatus() {
    try {
        const res = await fetch('/training/status');
        const data = await res.json();

        const statusMsg = document.getElementById('training-status-msg');
        statusMsg.textContent = data.message;

        if (data.status === 'complete' || data.status === 'failed') {
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;

            const btn = document.getElementById('train-btn');
            const btnText = document.getElementById('train-btn-text');
            const spinner = document.getElementById('train-spinner');

            btn.disabled = false;
            btnText.textContent = 'Train Model';
            spinner.classList.add('hidden');

            if (data.status === 'complete') {
                refreshStatus();
            } else {
                showError('Training failed: ' + data.message);
            }
        }
    } catch (e) {
        // Keep polling on transient errors
    }
}

// ===== Predictions =====
async function submitPrediction(event) {
    event.preventDefault();

    const input = document.getElementById('symbols-input').value.trim();
    if (!input) return;

    const symbols = input.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
    if (symbols.length === 0) return;

    const btn = document.getElementById('predict-btn');
    const btnText = document.getElementById('predict-btn-text');
    const spinner = document.getElementById('predict-spinner');

    btn.disabled = true;
    btnText.textContent = 'Loading...';
    spinner.classList.remove('hidden');

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const data = await res.json();
        renderResults(data);
    } catch (e) {
        showError(e.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Get Predictions';
        spinner.classList.add('hidden');
    }
}

function renderResults(data) {
    const section = document.getElementById('results-section');
    const meta = document.getElementById('results-meta');
    const grid = document.getElementById('results-grid');

    section.classList.remove('hidden');

    const modelInfo = data.model_name || 'Unknown';
    const trainedAt = data.model_last_trained
        ? new Date(data.model_last_trained).toLocaleString()
        : 'Unknown';
    meta.textContent = `Model: ${modelInfo} | Last trained: ${trainedAt}`;

    grid.innerHTML = data.predictions.map(p => {
        const dirClass = p.predicted_direction === 1 ? 'up' : 'down';
        const arrow = p.predicted_direction === 1 ? '\u2191' : '\u2193';
        const dirLabel = p.predicted_direction === 1 ? 'Up' : 'Down';
        const prob = (p.probability * 100).toFixed(1);
        const confClass = 'confidence-' + p.confidence;

        return `
            <div class="result-card">
                <div class="result-card-header">
                    <span class="result-symbol">${p.symbol}</span>
                    <span class="direction-arrow ${dirClass}" title="Predicted: ${dirLabel}">
                        ${arrow}
                    </span>
                </div>
                <div class="result-price">
                    Current: $${p.current_price.toFixed(2)}
                </div>
                <div class="result-details">
                    <span class="probability">${prob}%</span>
                    <span class="confidence-badge ${confClass}">${p.confidence}</span>
                </div>
            </div>
        `;
    }).join('');
}
