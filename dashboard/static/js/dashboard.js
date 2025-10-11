const COLORS = {
    primary: '#1A1A1A',
    red: '#E63946',
    blue: '#457B9D',
    yellow: '#F4A261',
    bg: '#FAFAFA'
};

let experimentsData = [];
let trainingData = [];

async function fetchData() {
    try {
        const [stats, experiments, training] = await Promise.all([
            fetch('/api/stats').then(r => r.json()),
            fetch('/api/experiments').then(r => r.json()),
            fetch('/api/training-logs').then(r => r.json())
        ]);
        
        updateStats(stats);
        experimentsData = experiments;
        trainingData = training;
        
        renderExperiments(experiments);
        renderCharts();
        renderTrainingTable();
    } catch (error) {
        console.error('Failed to fetch data:', error);
    }
}

function updateStats(stats) {
    document.getElementById('total-experiments').textContent = stats.total_experiments || 0;
    document.getElementById('train-accuracy').textContent = stats.avg_train_accuracy?.toFixed(2) || '—';
    document.getElementById('iid-accuracy').textContent = stats.avg_iid_accuracy?.toFixed(2) || '—';
    document.getElementById('compo-accuracy').textContent = stats.avg_compositional_accuracy?.toFixed(2) || '—';
}

function renderExperiments(experiments) {
    const container = document.getElementById('experiments-list');
    container.innerHTML = '';
    
    experiments.forEach(exp => {
        const card = document.createElement('div');
        const trainAcc = exp.metrics.train.acc;
        const category = trainAcc >= 0.8 ? 'high' : trainAcc >= 0.6 ? 'mid' : 'low';
        
        card.className = `experiment-card ${category}`;
        card.innerHTML = `
            <div class="experiment-id">${exp.experiment_id}</div>
            <div class="experiment-params">
                <div class="param"><span class="param-label">V:</span> ${exp.params.V}</div>
                <div class="param"><span class="param-label">Noise:</span> ${exp.params.channel_noise.toFixed(2)}</div>
                <div class="param"><span class="param-label">Length:</span> ${exp.params.length_cost.toFixed(2)}</div>
            </div>
            <div class="experiment-metrics">
                <div class="metric-row">
                    <span class="metric-label">Train</span>
                    <span class="metric-value">${trainAcc.toFixed(2)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">IID</span>
                    <span class="metric-value">${exp.metrics.iid.acc.toFixed(2)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Compositional</span>
                    <span class="metric-value">${exp.metrics.compo.acc.toFixed(2)}</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

function renderCharts() {
    if (trainingData.length > 0) {
        renderTrainingChart();
        renderLossChart();
    } else {
        showEmptyChart('training-chart');
        showEmptyChart('loss-chart');
    }
    
    if (experimentsData.length > 0) {
        renderAccuracyChart();
    } else {
        showEmptyChart('accuracy-chart');
    }
}

function showEmptyChart(canvasId) {
    const canvas = document.getElementById(canvasId);
    const container = canvas.parentElement;
    const placeholder = document.createElement('div');
    placeholder.className = 'empty-state';
    placeholder.innerHTML = `
        <div class="empty-state-icon">□</div>
        <div class="empty-state-text">No data available</div>
    `;
    canvas.style.display = 'none';
    container.appendChild(placeholder);
}

function renderTrainingChart() {
    const ctx = document.getElementById('training-chart');
    
    const steps = trainingData.map((d, i) => d.step || d.episode || i);
    const reward = trainingData.map(d => d.avg_reward || 0);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [{
                label: 'Avg Reward',
                data: reward,
                borderColor: COLORS.red,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 2,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { family: 'Space Grotesk', size: 11 } }
                },
                y: {
                    grid: { color: '#E0E0E0' },
                    ticks: { font: { family: 'Space Grotesk', size: 11 } }
                }
            }
        }
    });
}

function renderLossChart() {
    const ctx = document.getElementById('loss-chart');
    const steps = trainingData.map((d, i) => d.step || d.episode || i);
    const totalLoss = trainingData.map(d => d.total_loss || 0);
    const listenerLoss = trainingData.map(d => d.listener_loss || 0);
    const speakerLoss = trainingData.map(d => d.speaker_loss || 0);
    
    const datasets = [
        {
            label: 'Total Loss',
            data: totalLoss,
            borderColor: COLORS.primary,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0
        }
    ];
    
    if (listenerLoss.some(v => v !== 0)) {
        datasets.push({
            label: 'Listener Loss',
            data: listenerLoss,
            borderColor: COLORS.blue,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0
        });
    }
    
    if (speakerLoss.some(v => v !== 0)) {
        datasets.push({
            label: 'Speaker Loss',
            data: speakerLoss,
            borderColor: COLORS.yellow,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0
        });
    }
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { font: { family: 'Space Grotesk', size: 11 } }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { family: 'Space Grotesk', size: 11 } }
                },
                y: {
                    grid: { color: '#E0E0E0' },
                    ticks: { font: { family: 'Space Grotesk', size: 11 } }
                }
            }
        }
    });
}

function renderAccuracyChart() {
    const ctx = document.getElementById('accuracy-chart');
    const trainAccs = experimentsData.map(e => e.metrics.train.acc);
    const iidAccs = experimentsData.map(e => e.metrics.iid.acc);
    const compoAccs = experimentsData.map(e => e.metrics.compo.acc);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Train', 'IID', 'Compositional'],
            datasets: [{
                data: [
                    trainAccs.reduce((a, b) => a + b, 0) / trainAccs.length,
                    iidAccs.reduce((a, b) => a + b, 0) / iidAccs.length,
                    compoAccs.reduce((a, b) => a + b, 0) / compoAccs.length
                ],
                backgroundColor: [COLORS.red, COLORS.blue, COLORS.yellow],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { family: 'Space Grotesk', size: 11 } }
                },
                y: {
                    grid: { color: '#E0E0E0' },
                    ticks: { font: { family: 'Space Grotesk', size: 11 } },
                    beginAtZero: true
                }
            }
        }
    });
}

function renderTrainingTable() {
    const tbody = document.getElementById('metrics-tbody');
    tbody.innerHTML = '';
    
    if (trainingData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-secondary);">No training data available</td></tr>';
        return;
    }
    
    trainingData.slice(0, 50).forEach((row, i) => {
        const tr = document.createElement('tr');
        const step = row.step || row.episode || i;
        const reward = row.avg_reward || 0;
        const loss = row.total_loss || 0;
        const msgLen = row.avg_message_length || row.message_length || row.episode_length || 0;
        
        tr.innerHTML = `
            <td>${step}</td>
            <td>${reward.toFixed(3)}</td>
            <td>${loss.toFixed(3)}</td>
            <td>${msgLen.toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
    });
}

document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById(view).classList.add('active');
    });
});

document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const filter = btn.dataset.filter;
        
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        let filtered = experimentsData;
        if (filter === 'high') {
            filtered = experimentsData.filter(e => e.metrics.train.acc >= 0.8);
        } else if (filter === 'low') {
            filtered = experimentsData.filter(e => e.metrics.train.acc < 0.6);
        }
        
        renderExperiments(filtered);
    });
});

fetchData();

