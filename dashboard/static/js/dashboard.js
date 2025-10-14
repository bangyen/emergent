const COLORS = {
    primary: '#E63946',
    secondary: '#457B9D',
    tertiary: '#F4A261',
    background: '#F8F9FA',
    text: '#1D1D1F'
};

const CHART_CONFIG = {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 2,
    plugins: {
        legend: {
            display: true,
            position: 'bottom',
            labels: {
                font: { family: 'Space Grotesk', size: 12, weight: '500' },
                color: '#6E6E73',
                padding: 16,
                usePointStyle: true,
                pointStyle: 'rect'
            }
        }
    },
    scales: {
        x: {
            grid: { display: false, drawBorder: false },
            ticks: {
                font: { family: 'Space Grotesk', size: 11 },
                color: '#86868B'
            }
        },
        y: {
            grid: { color: '#E1E4E8', drawBorder: false },
            ticks: {
                font: { family: 'Space Grotesk', size: 11 },
                color: '#86868B'
            }
        }
    }
};

let charts = {};
let experimentsData = [];
let trainingData = [];

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-container');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const viewName = item.getAttribute('data-view');
            
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            views.forEach(v => v.classList.add('hidden'));
            document.getElementById(`${viewName}-view`).classList.remove('hidden');
            
            // Update page title based on view
            const titles = {
                'overview': 'Dashboard Overview',
                'experiments': 'Experiment Results',
                'training': 'Training Metrics'
            };
            document.querySelector('.page-title').textContent = titles[viewName] || 'Dashboard';
        });
    });
}

function initCharts() {
    // Training chart will be initialized when data loads
    // Accuracy chart will be initialized when data loads
    // Loss chart will be initialized when data loads
}

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
    
    const trainAcc = stats.avg_train_accuracy;
    const iidAcc = stats.avg_iid_accuracy;
    const compoAcc = stats.avg_compositional_accuracy;
    
    document.getElementById('train-accuracy').textContent = trainAcc ? trainAcc.toFixed(3) : '—';
    document.getElementById('iid-accuracy').textContent = iidAcc ? iidAcc.toFixed(3) : '—';
    document.getElementById('compo-accuracy').textContent = compoAcc ? compoAcc.toFixed(3) : '—';
}

function renderExperiments(experiments) {
    const container = document.getElementById('experiments-list');
    container.innerHTML = '';
    
    if (experiments.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">○</div><div class="empty-state-text">No experiments found</div></div>';
        return;
    }
    
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
                    <span class="metric-value">${trainAcc.toFixed(3)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">IID</span>
                    <span class="metric-value">${exp.metrics.iid.acc.toFixed(3)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Compositional</span>
                    <span class="metric-value">${exp.metrics.compo.acc.toFixed(3)}</span>
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
}

function showEmptyChart(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const container = canvas.parentElement;
    
    // Check if empty state already exists
    if (container.querySelector('.empty-state')) return;
    
    const placeholder = document.createElement('div');
    placeholder.className = 'empty-state';
    placeholder.innerHTML = `
        <div class="empty-state-icon">○</div>
        <div class="empty-state-text">No data available</div>
    `;
    canvas.style.display = 'none';
    container.appendChild(placeholder);
}

function renderTrainingChart() {
    const ctx = document.getElementById('training-chart');
    if (!ctx) return;
    
    // Clear any existing chart
    if (charts.training) {
        charts.training.destroy();
    }
    
    const steps = trainingData.map((d, i) => d.step || d.episode || i);
    const reward = trainingData.map(d => d.avg_reward || 0);
    
    charts.training = new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [{
                label: 'Avg Reward',
                data: reward,
                borderColor: COLORS.primary,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.1
            }]
        },
        options: CHART_CONFIG
    });
}

function renderLossChart() {
    const ctx = document.getElementById('loss-chart');
    if (!ctx) return;
    
    // Clear any existing chart
    if (charts.loss) {
        charts.loss.destroy();
    }
    
    const steps = trainingData.map((d, i) => d.step || d.episode || i);
    const totalLoss = trainingData.map(d => d.total_loss || 0);
    const listenerLoss = trainingData.map(d => d.listener_loss || 0);
    const speakerLoss = trainingData.map(d => d.speaker_loss || 0);
    
    const datasets = [
        {
            label: 'Total Loss',
            data: totalLoss,
            borderColor: COLORS.text,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4
        }
    ];
    
    if (listenerLoss.some(v => v !== 0)) {
        datasets.push({
            label: 'Listener Loss',
            data: listenerLoss,
            borderColor: COLORS.secondary,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4
        });
    }
    
    if (speakerLoss.some(v => v !== 0)) {
        datasets.push({
            label: 'Speaker Loss',
            data: speakerLoss,
            borderColor: COLORS.tertiary,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4
        });
    }
    
    charts.loss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: datasets
        },
        options: CHART_CONFIG
    });
}

function renderAccuracyChart() {
    const ctx = document.getElementById('accuracy-chart');
    if (!ctx) return;
    
    // Clear any existing chart
    if (charts.accuracy) {
        charts.accuracy.destroy();
    }
    
    const trainAccs = experimentsData.map(e => e.metrics.train.acc);
    const iidAccs = experimentsData.map(e => e.metrics.iid.acc);
    const compoAccs = experimentsData.map(e => e.metrics.compo.acc);
    
    charts.accuracy = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Train', 'IID', 'Compositional'],
            datasets: [{
                data: [
                    trainAccs.reduce((a, b) => a + b, 0) / trainAccs.length,
                    iidAccs.reduce((a, b) => a + b, 0) / iidAccs.length,
                    compoAccs.reduce((a, b) => a + b, 0) / compoAccs.length
                ],
                backgroundColor: [COLORS.primary, COLORS.secondary, COLORS.tertiary],
                borderWidth: 0
            }]
        },
        options: {
            ...CHART_CONFIG,
            aspectRatio: 1.8,
            plugins: {
                ...CHART_CONFIG.plugins,
                legend: { display: false }
            },
            scales: {
                ...CHART_CONFIG.scales,
                y: {
                    ...CHART_CONFIG.scales.y,
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function renderTrainingTable() {
    const tbody = document.getElementById('metrics-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (trainingData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="loading">No training data available</td></tr>';
        return;
    }
    
    trainingData.slice(0, 100).forEach((row, i) => {
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

function initFilterButtons() {
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
}

function initRefreshButton() {
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            // Add visual feedback
            refreshBtn.disabled = true;
            refreshBtn.style.opacity = '0.6';
            
            fetchData().then(() => {
                refreshBtn.disabled = false;
                refreshBtn.style.opacity = '1';
            });
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCharts();
    initFilterButtons();
    initRefreshButton();
    
    // Load initial data
    fetchData();
});
