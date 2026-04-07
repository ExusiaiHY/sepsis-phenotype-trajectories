/**
 * dashboard.js - Interactive visualization logic for ICU Sepsis Phenotype Dashboard
 */

// Global state
let summaryData = null;
let comparisonData = null;
let trajectoryData = null;
let transitionData = null;
let patientSamples = null;

// Color palette matching the backend
const COLORS = {
    phenotype: ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0'],
    methods: {
        PCA: '#2196F3',
        S1_masked: '#FF5722',
        S15_contrastive: '#4CAF50'
    }
};

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadAllData();
    initializeCharts();
    setupEventListeners();
});

async function loadAllData() {
    try {
        // Load all data in parallel
        const [summary, comparison, trajectory, transitions, patients] = await Promise.all([
            fetch('/api/summary').then(r => r.json()),
            fetch('/api/comparison').then(r => r.json()),
            fetch('/api/trajectory/stats').then(r => r.json()),
            fetch('/api/trajectory/transitions').then(r => r.json()),
            fetch('/api/patients/sample').then(r => r.json())
        ]);
        
        summaryData = summary;
        comparisonData = comparison;
        trajectoryData = trajectory;
        transitionData = transitions;
        patientSamples = patients;
        
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Failed to load data. Please refresh the page.');
    }
}

function showError(message) {
    alert(message);
}

// ============================================================
// Tab Navigation
// ============================================================

function showTab(tabId) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
    
    // Initialize tab-specific charts
    if (tabId === 'overview') {
        initializeOverviewCharts();
    } else if (tabId === 'comparison') {
        initializeComparisonCharts();
    } else if (tabId === 'trajectories') {
        initializeTrajectoryCharts();
    } else if (tabId === 'transitions') {
        initializeTransitionCharts();
    } else if (tabId === 'spatiotemporal') {
        initializeSpatioTemporal();
    } else if (tabId === 'patients') {
        initializePatientBrowser();
    }
}

// ============================================================
// Chart Initialization
// ============================================================

function initializeCharts() {
    // Initial charts for overview tab
    initializeOverviewCharts();
}

function initializeOverviewCharts() {
    if (!summaryData) return;
    
    // Update summary cards
    document.getElementById('total-patients').textContent = summaryData.n_patients.toLocaleString();
    document.getElementById('stable-pct').textContent = formatPercent(summaryData.stable_fraction);
    document.getElementById('single-pct').textContent = formatPercent(summaryData.single_transition_fraction);
    document.getElementById('multi-pct').textContent = formatPercent(summaryData.multi_transition_fraction);
    
    // Mortality by Phenotype Chart
    const mortalityCtx = document.getElementById('mortalityChart');
    if (mortalityCtx) {
        const mortData = summaryData.mortality_by_stable_phenotype;
        new Chart(mortalityCtx, {
            type: 'bar',
            data: {
                labels: ['P0 (Low)', 'P1 (High)', 'P2 (Highest)', 'P3 (Low-Mid)'],
                datasets: [{
                    label: 'Mortality Rate',
                    data: [
                        mortData.cluster_0.mortality_rate,
                        mortData.cluster_1.mortality_rate,
                        mortData.cluster_2.mortality_rate,
                        mortData.cluster_3.mortality_rate
                    ],
                    backgroundColor: COLORS.phenotype,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Mortality: ${formatPercent(ctx.raw)}`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 0.35,
                        ticks: {
                            callback: (val) => formatPercent(val)
                        }
                    }
                }
            }
        });
    }
    
    // Method Comparison Chart
    const methodCtx = document.getElementById('methodComparisonChart');
    if (methodCtx && summaryData.method_comparison) {
        const mc = summaryData.method_comparison;
        new Chart(methodCtx, {
            type: 'radar',
            data: {
                labels: ['Silhouette Score', 'Clinical Separation (Mortality Range)'],
                datasets: [
                    {
                        label: 'PCA',
                        data: [mc.PCA.silhouette * 3, mc.PCA.mort_range],
                        borderColor: COLORS.methods.PCA,
                        backgroundColor: COLORS.methods.PCA + '40'
                    },
                    {
                        label: 'S1 (Masked)',
                        data: [mc.S1_masked.silhouette * 3, mc.S1_masked.mort_range],
                        borderColor: COLORS.methods.S1_masked,
                        backgroundColor: COLORS.methods.S1_masked + '40'
                    },
                    {
                        label: 'S1.5 (Contrastive)',
                        data: [mc.S15_contrastive.silhouette * 3, mc.S15_contrastive.mort_range],
                        borderColor: COLORS.methods.S15_contrastive,
                        backgroundColor: COLORS.methods.S15_contrastive + '40'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 0.35
                    }
                }
            }
        });
    }
}

function initializeComparisonCharts() {
    if (!comparisonData) return;
    
    // Silhouette Comparison
    const silCtx = document.getElementById('silhouetteChart');
    if (silCtx) {
        const methods = ['PCA_baseline', 'S1_masked', 'S15_contrastive'];
        const methodLabels = ['PCA', 'S1 (Masked)', 'S1.5 (Contrastive)'];
        
        new Chart(silCtx, {
            type: 'bar',
            data: {
                labels: methodLabels,
                datasets: [
                    {
                        label: 'K=2',
                        data: methods.map(m => comparisonData[m]['K=2'].aggregated.silhouette.mean),
                        backgroundColor: '#90CAF9'
                    },
                    {
                        label: 'K=4',
                        data: methods.map(m => comparisonData[m]['K=4'].aggregated.silhouette.mean),
                        backgroundColor: '#1565C0'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Silhouette Score' }
                    }
                }
            }
        });
    }
    
    // Mortality Range Comparison
    const mortCtx = document.getElementById('mortRangeChart');
    if (mortCtx) {
        const methods = ['PCA_baseline', 'S1_masked', 'S15_contrastive'];
        const methodLabels = ['PCA', 'S1 (Masked)', 'S1.5 (Contrastive)'];
        
        new Chart(mortCtx, {
            type: 'bar',
            data: {
                labels: methodLabels,
                datasets: [
                    {
                        label: 'K=2',
                        data: methods.map(m => comparisonData[m]['K=2'].aggregated.mort_range.mean),
                        backgroundColor: '#FFCC80'
                    },
                    {
                        label: 'K=4',
                        data: methods.map(m => comparisonData[m]['K=4'].aggregated.mort_range.mean),
                        backgroundColor: '#EF6C00'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${formatPercent(ctx.raw)}`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 0.35,
                        ticks: {
                            callback: (val) => formatPercent(val)
                        }
                    }
                }
            }
        });
    }
    
    // Populate comparison table
    populateComparisonTable();
}

function populateComparisonTable() {
    const tbody = document.getElementById('comparison-table-body');
    if (!tbody || !comparisonData) return;
    
    const methods = [
        { key: 'PCA_baseline', label: 'PCA Baseline' },
        { key: 'S1_masked', label: 'S1 (Masked)' },
        { key: 'S15_contrastive', label: 'S1.5 (Contrastive)' }
    ];
    
    tbody.innerHTML = methods.map(m => {
        const k2 = comparisonData[m.key]['K=2'].aggregated;
        const k4 = comparisonData[m.key]['K=4'].aggregated;
        return `
            <tr class="hover:bg-gray-50">
                <td class="px-4 py-3 font-medium">${m.label}</td>
                <td class="px-4 py-3 text-right">${k2.silhouette.mean.toFixed(4)}</td>
                <td class="px-4 py-3 text-right">${formatPercent(k2.mort_range.mean)}</td>
                <td class="px-4 py-3 text-right font-medium">${k4.silhouette.mean.toFixed(4)}</td>
                <td class="px-4 py-3 text-right font-medium">${formatPercent(k4.mort_range.mean)}</td>
            </tr>
        `;
    }).join('');
}

function initializeTrajectoryCharts() {
    if (!trajectoryData) return;
    
    // Window prevalence chart
    const prevCtx = document.getElementById('prevalenceChart');
    if (prevCtx) {
        // Calculate prevalence from trajectory data
        const windowLabels = trajectoryData.patient_level.top_trajectory_patterns
            .filter(p => p.count > 100); // Filter significant patterns
        
        // Create a simple visualization of phenotype distribution over time
        const datasets = [0, 1, 2, 3].map((phenotype, idx) => ({
            label: `Phenotype ${phenotype}`,
            data: [0.25, 0.25, 0.25, 0.25], // Placeholder - would need actual window data
            backgroundColor: COLORS.phenotype[idx],
            stack: 'stack1'
        }));
        
        new Chart(prevCtx, {
            type: 'bar',
            data: {
                labels: ['W0 [0,24)', 'W1 [6,30)', 'W2 [12,36)', 'W3 [18,42)', 'W4 [24,48)'],
                datasets: datasets
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    x: { stacked: true },
                    y: { 
                        stacked: true,
                        max: 1,
                        ticks: { callback: (val) => formatPercent(val) }
                    }
                }
            }
        });
    }
    
    // Top trajectory patterns
    const patternsDiv = document.getElementById('trajectory-patterns');
    if (patternsDiv) {
        const patterns = trajectoryData.patient_level.top_trajectory_patterns.slice(0, 15);
        patternsDiv.innerHTML = patterns.map(p => {
            const patternStr = p.pattern.map(x => `P${x}`).join(' → ');
            const barWidth = (p.fraction * 100).toFixed(1);
            return `
                <div class="flex items-center">
                    <div class="w-48 text-sm font-mono">${patternStr}</div>
                    <div class="flex-1 mx-4">
                        <div class="h-6 bg-gray-200 rounded-full overflow-hidden">
                            <div class="h-full bg-blue-500 rounded-full" style="width: ${barWidth}%"></div>
                        </div>
                    </div>
                    <div class="w-24 text-right text-sm">
                        ${(p.fraction * 100).toFixed(1)}% (${p.count.toLocaleString()})
                    </div>
                </div>
            `;
        }).join('');
    }
}

function initializeTransitionCharts() {
    if (!transitionData) return;
    
    // Update statistics
    document.getElementById('total-transitions').textContent = transitionData.total_transition_events.toLocaleString();
    document.getElementById('self-transitions').textContent = transitionData.self_transition_events.toLocaleString();
    document.getElementById('non-self-fraction').textContent = formatPercent(transitionData.non_self_fraction);
    
    // Transition heatmap using Plotly
    const heatmapDiv = document.getElementById('transition-heatmap');
    if (heatmapDiv) {
        const z = transitionData.transition_prob_matrix;
        const data = [{
            z: z,
            x: ['P0', 'P1', 'P2', 'P3'],
            y: ['P0', 'P1', 'P2', 'P3'],
            type: 'heatmap',
            colorscale: 'Blues',
            text: z.map(row => row.map(v => (v * 100).toFixed(1) + '%')),
            texttemplate: '%{text}',
            textfont: { size: 14 }
        }];
        
        const layout = {
            xaxis: { title: 'To Phenotype' },
            yaxis: { title: 'From Phenotype', autorange: 'reversed' },
            annotations: []
        };
        
        Plotly.newPlot('transition-heatmap', data, layout, {responsive: true});
    }
    
    // Top non-self transitions
    const topDiv = document.getElementById('top-transitions');
    if (topDiv) {
        const transitions = transitionData.top_non_self_transitions.slice(0, 10);
        topDiv.innerHTML = transitions.map((t, i) => `
            <div class="flex items-center p-3 bg-gray-50 rounded-lg">
                <div class="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-sm mr-3">
                    ${i + 1}
                </div>
                <div class="flex-1">
                    <div class="flex items-center">
                        <span class="px-2 py-1 rounded text-xs font-medium" 
                              style="background-color: ${COLORS.phenotype[t.from]}">
                            P${t.from}
                        </span>
                        <span class="mx-2 text-gray-400">→</span>
                        <span class="px-2 py-1 rounded text-xs font-medium" 
                              style="background-color: ${COLORS.phenotype[t.to]}">
                            P${t.to}
                        </span>
                    </div>
                </div>
                <div class="text-right">
                    <div class="font-semibold">${t.count.toLocaleString()}</div>
                    <div class="text-xs text-gray-500">${(t.prob * 100).toFixed(1)}%</div>
                </div>
            </div>
        `).join('');
    }
}

// Global state for spatio-temporal
let stProjectionData = null;
let stEvolutionData = null;

function initializePatientBrowser() {
    if (!patientSamples) return;
    
    // Update stats
    document.getElementById('browser-total').textContent = patientSamples.total_patients.toLocaleString();
    document.getElementById('browser-stable').textContent = patientSamples.categories.stable.toLocaleString();
    document.getElementById('browser-single').textContent = patientSamples.categories.single_transition.toLocaleString();
    document.getElementById('browser-multi').textContent = patientSamples.categories.multi_transition.toLocaleString();
    
    // Load initial patient list
    loadPatientList();
}

// ============================================================
// Spatio-Temporal Visualization Functions
// ============================================================

async function initializeSpatioTemporal() {
    await loadSpatioTemporalData();
    renderEmbeddingProjection();
    renderMovementChart();
    renderNormChart();
    renderEvolutionStats();
    populateSTPatientSelect();
}

async function loadSpatioTemporalData() {
    try {
        const [projection, evolution] = await Promise.all([
            fetch('/api/spatiotemporal/projection').then(r => r.json()),
            fetch('/api/spatiotemporal/window-evolution').then(r => r.json())
        ]);
        
        stProjectionData = projection;
        stEvolutionData = evolution;
    } catch (error) {
        console.error('Error loading spatio-temporal data:', error);
    }
}

function renderEmbeddingProjection() {
    if (!stProjectionData) return;
    
    const div = document.getElementById('embedding-projection');
    if (!div) return;
    
    const { coords_2d, labels, patient_ids, window_indices } = stProjectionData;
    
    // Create separate traces for each phenotype
    const traces = [0, 1, 2, 3].map(p => {
        const indices = labels.map((l, i) => l === p ? i : -1).filter(i => i !== -1);
        return {
            x: indices.map(i => coords_2d[i][0]),
            y: indices.map(i => coords_2d[i][1]),
            mode: 'markers',
            type: 'scatter',
            name: `Phenotype ${p}`,
            marker: {
                size: 6,
                color: COLORS.phenotype[p],
                opacity: 0.7
            },
            text: indices.map(i => `Patient ${patient_ids[i]}, Window ${window_indices[i]}`),
            hovertemplate: '%{text}<extra></extra>'
        };
    });
    
    const layout = {
        xaxis: { title: 'PCA Component 1' },
        yaxis: { title: 'PCA Component 2' },
        hovermode: 'closest',
        legend: { x: 0, y: 1 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot('embedding-projection', traces, layout, {responsive: true});
}

function renderMovementChart() {
    if (!stEvolutionData) return;
    
    const div = document.getElementById('movement-chart');
    if (!div) return;
    
    const movements = stEvolutionData.embedding_movements;
    
    const data = [{
        x: movements.map(m => `W${m.from_window}→W${m.to_window}`),
        y: movements.map(m => m.mean_distance),
        type: 'bar',
        marker: {
            color: movements.map((_, i) => `rgba(147, 51, 234, ${0.5 + i * 0.1})`),
            line: { color: '#9333ea', width: 2 }
        },
        text: movements.map(m => m.mean_distance.toFixed(3)),
        textposition: 'auto'
    }];
    
    const layout = {
        xaxis: { title: 'Window Transition' },
        yaxis: { title: 'Mean L2 Distance' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 20, b: 40 }
    };
    
    Plotly.newPlot('movement-chart', data, layout, {responsive: true});
}

function renderNormChart() {
    if (!stEvolutionData) return;
    
    const div = document.getElementById('norm-chart');
    if (!div) return;
    
    const norms = stEvolutionData.phenotype_embedding_norms;
    
    const data = [{
        x: ['P0', 'P1', 'P2', 'P3'],
        y: [norms[0].mean, norms[1].mean, norms[2].mean, norms[3].mean],
        type: 'bar',
        marker: {
            color: COLORS.phenotype,
            line: { width: 2, color: '#333' }
        },
        error_y: {
            type: 'data',
            array: [norms[0].std, norms[1].std, norms[2].std, norms[3].std],
            visible: true
        },
        text: [norms[0].mean.toFixed(2), norms[1].mean.toFixed(2), norms[2].mean.toFixed(2), norms[3].mean.toFixed(2)],
        textposition: 'auto'
    }];
    
    const layout = {
        xaxis: { title: 'Phenotype' },
        yaxis: { title: 'Embedding Norm (Mean ± Std)' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 20, b: 40 }
    };
    
    Plotly.newPlot('norm-chart', data, layout, {responsive: true});
}

function renderEvolutionStats() {
    if (!stEvolutionData) return;
    
    const tbody = document.getElementById('evolution-stats-body');
    if (!tbody) return;
    
    const movements = stEvolutionData.embedding_movements;
    
    tbody.innerHTML = movements.map(m => {
        let interpretation = '';
        if (m.mean_distance > 8) interpretation = 'High variability period';
        else if (m.mean_distance > 6) interpretation = 'Moderate state changes';
        else interpretation = 'Stable period';
        
        return `
            <tr class="hover:bg-gray-50">
                <td class="px-4 py-3 font-medium">Window ${m.from_window} → ${m.to_window}</td>
                <td class="px-4 py-3 text-right">${m.mean_distance.toFixed(3)}</td>
                <td class="px-4 py-3 text-right">${m.std_distance.toFixed(3)}</td>
                <td class="px-4 py-3 text-right text-sm text-gray-600">${interpretation}</td>
            </tr>
        `;
    }).join('');
}

function populateSTPatientSelect() {
    const select = document.getElementById('st-patient-select');
    if (!select || !stProjectionData) return;
    
    const sampleIds = stProjectionData.sample_patient_ids;
    select.innerHTML = `
        <option value="">Select a patient...</option>
        ${sampleIds.map(id => `<option value="${id}">Patient ${id}</option>`).join('')}
    `;
}

async function loadSTPatientTrajectory() {
    const patientId = document.getElementById('st-patient-select').value;
    if (!patientId) return;
    
    try {
        const response = await fetch(`/api/spatiotemporal/embedding-trajectory/${patientId}`);
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // Project 128D trajectory to 3D using PCA
        const trajectory = data.embedding_trajectory;  // 5 × 128
        
        // Simple manual PCA to 3D
        const centered = trajectory.map(t => t.map((v, i) => v - trajectory[0][i]));
        
        // Use first 3 principal directions (simplified)
        const coords3d = centered.map((t, idx) => ({
            x: t.slice(0, 42).reduce((a, b) => a + b, 0) / 42,
            y: t.slice(42, 84).reduce((a, b) => a + b, 0) / 42,
            z: t.slice(84, 128).reduce((a, b) => a + b, 0) / 44,
            label: `W${idx} (P${data.window_labels[idx]})`
        }));
        
        // Create 3D scatter with connecting line
        const scatterTrace = {
            x: coords3d.map(c => c.x),
            y: coords3d.map(c => c.y),
            z: coords3d.map(c => c.z),
            mode: 'markers+lines+text',
            type: 'scatter3d',
            name: 'Trajectory',
            marker: {
                size: 10,
                color: coords3d.map((c, i) => COLORS.phenotype[data.window_labels[i]]),
                line: { color: '#333', width: 2 }
            },
            line: { color: '#666', width: 3 },
            text: coords3d.map(c => c.label),
            textposition: 'top center',
            textfont: { size: 10 }
        };
        
        const layout = {
            scene: {
                xaxis: { title: 'PC1 (approx)' },
                yaxis: { title: 'PC2 (approx)' },
                zaxis: { title: 'PC3 (approx)' }
            },
            margin: { t: 20, b: 20 },
            paper_bgcolor: 'rgba(0,0,0,0)'
        };
        
        Plotly.newPlot('individual-trajectory', [scatterTrace], layout, {responsive: true});
        
    } catch (error) {
        console.error('Error loading trajectory:', error);
        alert('Failed to load patient trajectory');
    }
}

function refreshProjection() {
    // Reload projection with new random sample
    loadSpatioTemporalData().then(() => {
        renderEmbeddingProjection();
        populateSTPatientSelect();
    });
}

function toggleTrajectories() {
    // TODO: Add trajectory line overlay on projection
    console.log('Toggle trajectories');
}

// ============================================================
// Patient Browser Functions
// ============================================================

function loadPatientList() {
    const category = document.getElementById('patient-category').value;
    const select = document.getElementById('patient-select');
    
    if (!patientSamples) return;
    
    const indices = patientSamples.sample_indices[category];
    select.innerHTML = indices.map(id => 
        `<option value="${id}">Patient ${id}</option>`
    ).join('');
}

async function loadPatientDetail() {
    const patientId = document.getElementById('patient-select').value;
    const container = document.getElementById('patient-detail-content');
    
    try {
        const response = await fetch(`/api/patient/${patientId}`);
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<p class="text-red-500">${data.error}</p>`;
            return;
        }
        
        // Create trajectory visualization
        const trajectoryViz = data.trajectory.map((p, i) => `
            <div class="flex flex-col items-center">
                <div class="text-xs text-gray-500 mb-1">${data.window_starts[i]}h</div>
                <div class="w-16 h-16 rounded-full flex items-center justify-center text-white font-bold text-xl"
                     style="background-color: ${COLORS.phenotype[p]}">
                    P${p}
                </div>
            </div>
            ${i < data.trajectory.length - 1 ? `
                <div class="flex items-center mx-2">
                    <div class="text-2xl text-gray-400">→</div>
                </div>
            ` : ''}
        `).join('');
        
        container.innerHTML = `
            <div class="space-y-6">
                <div class="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div>
                        <div class="text-sm text-gray-600">Patient ID</div>
                        <div class="text-xl font-bold">${data.patient_id}</div>
                    </div>
                    <div>
                        <div class="text-sm text-gray-600">Trajectory Type</div>
                        <span class="px-3 py-1 rounded-full text-sm font-medium ${getTrajectoryTypeClass(data.trajectory_type)}">
                            ${data.trajectory_type.replace('_', ' ').toUpperCase()}
                        </span>
                    </div>
                    <div>
                        <div class="text-sm text-gray-600">Unique Phenotypes</div>
                        <div class="text-xl font-bold">${data.unique_phenotypes.length}</div>
                    </div>
                </div>
                
                <div>
                    <h4 class="font-medium mb-4">Phenotype Trajectory</h4>
                    <div class="flex items-center justify-center p-6 bg-gray-50 rounded-lg overflow-x-auto">
                        ${trajectoryViz}
                    </div>
                </div>
                
                <div class="grid grid-cols-4 gap-4">
                    ${[0, 1, 2, 3].map(p => {
                        const count = data.trajectory.filter(x => x === p).length;
                        const percent = (count / data.trajectory.length * 100).toFixed(0);
                        return `
                            <div class="p-4 rounded-lg text-center" style="background-color: ${COLORS.phenotype[p]}20">
                                <div class="w-8 h-8 rounded mx-auto mb-2" style="background-color: ${COLORS.phenotype[p]}"></div>
                                <div class="text-sm font-medium">Phenotype ${p}</div>
                                <div class="text-2xl font-bold">${percent}%</div>
                                <div class="text-xs text-gray-600">${count} windows</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
        
    } catch (error) {
        container.innerHTML = `<p class="text-red-500">Error loading patient data</p>`;
    }
}

function getTrajectoryTypeClass(type) {
    const classes = {
        'stable': 'bg-green-100 text-green-800',
        'single_transition': 'bg-orange-100 text-orange-800',
        'multi_transition': 'bg-purple-100 text-purple-800'
    };
    return classes[type] || 'bg-gray-100';
}

// ============================================================
// Utility Functions
// ============================================================

function formatPercent(value) {
    if (value === null || value === undefined) return '-';
    return (value * 100).toFixed(1) + '%';
}

function setupEventListeners() {
    // Any additional event listeners
}
