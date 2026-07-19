const state = {
    sources: [], raw: [], datasets: [], models: [], jobs: [],
    history: [], layers: {}, activeLayer: 'reflectivity',
    frames: [], labels: [], frameIndex: 0, playerTimer: null,
};

const paneMeta = {
    overview: ['Обзор', 'Состояние системы и последние операции'],
    forecast: ['Прогноз', 'Наблюдения, перенос и эволюция радиоэха'],
    sources: ['Источники', 'Количественные, discovery и визуальные контуры'],
    data: ['Архив', 'Загрузка и canonical-предобработка'],
    training: ['Обучение', 'Выборка, балансировка и архитектура'],
    jobs: ['Задания', 'Очередь и журналы локального worker'],
    models: ['Модели', 'Реестр, метрики и выбор рабочей модели'],
};
const layerNames = {
    reflectivity: 'Отражаемость', motion: 'Перенос', growth: 'Рост',
    decay: 'Распад', uncertainty: 'Неопределённость',
};
const byId = id => document.getElementById(id);

async function api(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.error || data.success === false) {
        throw new Error(data.error || data.message || `HTTP ${response.status}`);
    }
    return data;
}
function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>'"]/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
    })[character]);
}
function showToast(message, error = false) {
    const element = byId('toast');
    element.textContent = message;
    element.className = `toast show${error ? ' error' : ''}`;
    clearTimeout(element.hideTimer);
    element.hideTimer = setTimeout(() => element.className = 'toast', 3200);
}
function empty(text) { return `<div class="empty">${escapeHtml(text)}</div>`; }
function formatDate(value) {
    return value ? new Date(value).toLocaleString('ru-RU', {dateStyle: 'short', timeStyle: 'short'}) : '—';
}
function statusBadge(status) {
    const className = ['completed', 'published'].includes(status) ? 'good'
        : ['failed', 'interrupted'].includes(status) ? 'bad'
        : ['running', 'queued', 'cancelling'].includes(status) ? 'info' : 'warn';
    return `<span class="badge ${className}">${escapeHtml(status || 'unknown')}</span>`;
}
function classSummary(counts = {}) {
    const text = Object.entries(counts).filter(([, value]) => value).map(([name, value]) => `${name}: ${value}`).join(' · ');
    return text || 'классы ещё не рассчитаны';
}

function showPane(name) {
    document.querySelectorAll('.pane').forEach(element => {
        element.classList.toggle('active', element.id === `pane-${name}`);
    });
    document.querySelectorAll('[data-pane]').forEach(element => {
        element.classList.toggle('active', element.dataset.pane === name);
    });
    byId('pageTitle').textContent = paneMeta[name][0];
    byId('pageSubtitle').textContent = paneMeta[name][1];
    window.scrollTo({top: 0, behavior: 'smooth'});
}

function renderSources() {
    const cards = state.sources.map(source => `
        <div class="card source-card"><div class="card-body">
            <div class="symbol">${source.training_allowed ? '◉' : source.visualization_allowed ? '◌' : '⌕'}</div>
            <h3>${escapeHtml(source.source_id)}</h3>
            <p>${escapeHtml(source.notes || source.native_format)}</p>
            <div class="source-flags">
                <span class="badge ${source.training_allowed ? 'good' : 'warn'}">${source.training_allowed ? 'обучение' : 'не для train'}</span>
                <span class="badge">${escapeHtml(source.native_format)}</span>
            </div>
        </div></div>`).join('');
    byId('sourcesGrid').innerHTML = cards || empty('Источники не зарегистрированы');
    byId('overviewSources').innerHTML = state.sources.length ? state.sources.slice(0, 3).map(source => `
        <div class="card"><div class="card-body"><strong>${escapeHtml(source.source_id)}</strong>
        <p style="color:var(--muted);font-size:13px">${escapeHtml(source.native_format)}</p>
        <span class="badge ${source.training_allowed ? 'good' : 'warn'}">${source.training_allowed ? 'quantitative' : 'restricted'}</span>
        </div></div>`).join('') : empty('Нет данных');
}

function renderRaw() {
    byId('prepareArchive').innerHTML = '<option value="">Выберите архив</option>' + state.raw
        .filter(item => item.status === 'completed')
        .map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.folder_name)} · ${escapeHtml(item.station || '')}</option>`)
        .join('');
    byId('rawList').innerHTML = state.raw.length ? state.raw.map(item => `
        <div class="list-item"><div class="meta"><strong>${escapeHtml(item.folder_name)}</strong>
        <small>${escapeHtml(item.station || '—')} · ${escapeHtml(item.date || '')} · ${item.downloaded_count || 0} файлов</small></div>
        <div class="row-actions">${statusBadge(item.status)}
        <button class="button mini" data-action="choose-archive" data-path="${escapeHtml(item.path)}">Подготовить</button></div></div>`).join('') : empty('Архив пока пуст');
}

function renderDatasets() {
    const completed = state.datasets.filter(item => item.status === 'completed');
    byId('datasetPicker').innerHTML = completed.length ? completed.map(item => `
        <label class="dataset-option"><input type="checkbox" class="dataset-check" value="${escapeHtml(item.path)}">
        <span><strong>${escapeHtml(item.folder_name)}</strong><small style="display:block;color:var(--muted)">
        ${item.sample_count || 0} sequences · ${escapeHtml(item.pipeline?.pipeline_version || '')}<br>
        ${escapeHtml(classSummary(item.class_counts))}</small></span></label>`).join('') : empty('Сначала подготовьте датасет');
    byId('datasetStats').innerHTML = state.datasets.length ? state.datasets.map(item => `
        <div class="card"><div class="card-body"><strong>${escapeHtml(item.folder_name)}</strong>
        <p style="color:var(--muted);font-size:13px">${escapeHtml(classSummary(item.class_counts))}</p>
        <span class="badge info">${item.sample_count || 0} samples</span></div></div>`).join('') : empty('Нет статистики');
}

function renderModels() {
    byId('metricModels').textContent = state.models.length;
    byId('modelsGrid').innerHTML = state.models.length ? state.models.map(item => `
        <div class="card"><div class="card-head"><h3>${escapeHtml(item.folder_name)}</h3>${statusBadge(item.status)}</div>
        <div class="card-body"><div class="source-flags">
        <span class="badge">${escapeHtml(item.model_architecture || 'unknown')}</span>
        <span class="badge info">${escapeHtml(item.pipeline_version || '')}</span>
        <span class="badge">${item.horizon_minutes || '—'} мин</span></div>
        <p style="color:var(--muted);font-size:13px">Loss: ${item.metrics?.best_val_loss != null ? Number(item.metrics.best_val_loss).toFixed(6) : '—'} · ${escapeHtml(item.sampling || 'natural')}</p>
        <div class="row-actions"><button class="button mini" data-action="model-details" data-id="${escapeHtml(item.folder_name)}">Метаданные</button>
        <button class="button mini primary" data-action="load-model" data-path="${escapeHtml(item.path)}" data-name="${escapeHtml(item.folder_name)}" ${item.usable ? '' : 'disabled'}>Использовать</button></div>
        </div></div>`).join('') : empty('Реестр моделей пуст');
}

function jobMarkup(job) {
    const cancellable = ['queued', 'running', 'cancelling'].includes(job.status);
    return `<div class="list-item job-line status-${escapeHtml(job.status)}">
        <div><span class="progress-dot"></span> <strong>${escapeHtml(job.kind)}</strong></div>
        <div class="meta"><small>${formatDate(job.created_at)}${job.error ? ' · ' + escapeHtml(job.error) : ''}</small></div>
        <div class="row-actions">${statusBadge(job.status)}
        <button class="button mini" data-action="job-log" data-id="${job.id}">Журнал</button>
        ${cancellable ? `<button class="button mini danger" data-action="cancel-job" data-id="${job.id}">Отмена</button>` : ''}</div></div>`;
}
function renderJobs() {
    byId('jobsList').innerHTML = state.jobs.length ? state.jobs.map(jobMarkup).join('') : empty('Заданий пока нет');
    byId('recentJobs').innerHTML = state.jobs.length ? state.jobs.slice(0, 5).map(jobMarkup).join('') : empty('Заданий пока нет');
}
function renderMetrics() {
    byId('metricRaw').textContent = state.raw.length;
    byId('metricDatasets').textContent = state.datasets.length;
}

async function refreshAll(silent = false) {
    try {
        const [sources, raw, datasets, models, jobs] = await Promise.all([
            api('/api/sources'), api('/api/inventory/raw'), api('/api/inventory/datasets'),
            api('/api/inventory/models'), api('/api/jobs'),
        ]);
        Object.assign(state, {sources, raw, datasets, models, jobs});
        renderSources(); renderRaw(); renderDatasets(); renderModels(); renderJobs(); renderMetrics();
        if (!silent) showToast('Данные обновлены');
    } catch (error) { showToast(error.message, true); }
}
async function refreshJobsOnly() {
    try { state.jobs = await api('/api/jobs'); renderJobs(); } catch (_) { /* next polling cycle */ }
}

async function submitJob(form, url) {
    const data = await api(url, {method: 'POST', body: new FormData(form)});
    showToast(`Задание ${data.task_id.slice(0, 8)} поставлено в очередь`);
    await refreshAll(true);
    showPane('jobs');
}
async function openLog(id) {
    const data = await api(`/api/task/logs/${id}`);
    byId('logTitle').textContent = `${data.kind} · ${data.status}`;
    byId('logContent').textContent = data.logs || 'Журнал пока пуст';
    byId('logDialog').showModal();
}
async function cancelJob(id) {
    await api(`/api/jobs/${id}/cancel`, {method: 'POST'});
    showToast('Запрошена отмена');
    await refreshJobsOnly();
}
async function loadModel(path, name) {
    const body = new FormData(); body.append('model_path', path);
    const data = await api('/api/model/load', {method: 'POST', body});
    byId('activeModel').textContent = `${name} · ${data.model.model_architecture}`;
    showToast('Модель загружена');
}
async function showModel(id) {
    const data = await api(`/api/model/details/${id}`);
    byId('logTitle').textContent = `Модель ${id}`;
    byId('logContent').textContent = JSON.stringify(data.metadata, null, 2);
    byId('logDialog').showModal();
}

function stopPlayer() {
    clearInterval(state.playerTimer); state.playerTimer = null; byId('playBtn').textContent = '▶';
}
function setFrame(index) {
    if (!state.frames.length) return;
    state.frameIndex = Math.max(0, Math.min(index, state.frames.length - 1));
    byId('viewerImage').src = state.frames[state.frameIndex];
    byId('viewerImage').style.display = 'block';
    byId('viewerPlaceholder').style.display = 'none';
    byId('frameSlider').value = state.frameIndex;
    byId('frameLabel').textContent = state.labels[state.frameIndex];
}
function selectLayer(name) {
    stopPlayer(); state.activeLayer = name;
    const items = state.layers[name] || [];
    const combined = [...state.history, ...items];
    state.frames = combined.map(item => `data:image/png;base64,${item.data}`);
    state.labels = combined.map(item => item.label);
    byId('frameSlider').max = Math.max(state.frames.length - 1, 0);
    byId('frameSlider').disabled = !state.frames.length;
    byId('playBtn').disabled = !state.frames.length;
    const initial = name === 'reflectivity' ? Math.max(state.history.length - 1, 0) : Math.min(state.history.length, state.frames.length - 1);
    setFrame(initial);
}
function configureLayers(layers) {
    state.layers = layers || {};
    const names = Object.keys(state.layers);
    byId('layerSelect').innerHTML = names.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(layerNames[name] || name)}</option>`).join('');
    byId('layerSelect').disabled = names.length < 2;
    selectLayer(names.includes('reflectivity') ? 'reflectivity' : names[0]);
}

async function submitForecast(form) {
    stopPlayer();
    const data = await api('/api/predict', {method: 'POST', body: new FormData(form)});
    state.history = data.history || [];
    configureLayers(data.layers || {reflectivity: data.forecast || []});
    byId('exportBtn').disabled = false;
    const evolution = data.evolution_diagnostics || {};
    byId('diagnostics').innerHTML = `
        <span class="badge info">${escapeHtml(data.model_architecture)}</span>
        <span class="badge">${escapeHtml(data.pipeline_version)}</span>
        <span class="badge">${data.grid?.width || '—'}×${data.grid?.height || '—'}</span>
        <span class="badge">max ${Number(data.diagnostics.max_dbz).toFixed(1)} dBZ</span>
        ${evolution.mean_motion_pixels != null ? `<span class="badge">motion ${Number(evolution.mean_motion_pixels).toFixed(2)} px</span>` : ''}
        <span class="badge warn">экспериментальный продукт</span>`;
}

function syncSourceFields() {
    const value = byId('forecastSource').value;
    byId('localField').hidden = value !== 'local';
    byId('uploadField').hidden = value !== 'upload';
    byId('stationField').hidden = !['aws', 'ftp'].includes(value);
}
async function loadStations() {
    try {
        const stations = await api('/api/ftp/stations');
        if (!stations.length) return;
        byId('forecastStation').innerHTML = stations.map(station => `<option value="${escapeHtml(station.code)}">${escapeHtml(station.code.toUpperCase())} · ${escapeHtml(station.name)}</option>`).join('');
        byId('downloadStation').innerHTML = stations.map(station => `<option value="${escapeHtml(station.code.toUpperCase())}">${escapeHtml(station.code.toUpperCase())} · ${escapeHtml(station.name)}</option>`).join('');
    } catch (_) { /* defaults remain available */ }
}

function handleAction(button) {
    const action = button.dataset.action;
    if (action === 'choose-archive') {
        byId('prepareArchive').value = button.dataset.path; showPane('data'); byId('prepareArchive').focus();
    } else if (action === 'job-log') {
        openLog(button.dataset.id).catch(error => showToast(error.message, true));
    } else if (action === 'cancel-job') {
        cancelJob(button.dataset.id).catch(error => showToast(error.message, true));
    } else if (action === 'load-model') {
        loadModel(button.dataset.path, button.dataset.name).catch(error => showToast(error.message, true));
    } else if (action === 'model-details') {
        showModel(button.dataset.id).catch(error => showToast(error.message, true));
    }
}

function init() {
    document.querySelectorAll('[data-pane]').forEach(button => button.addEventListener('click', () => showPane(button.dataset.pane)));
    document.querySelectorAll('[data-go]').forEach(button => button.addEventListener('click', () => showPane(button.dataset.go)));
    document.addEventListener('click', event => { const button = event.target.closest('[data-action]'); if (button) handleAction(button); });
    byId('themeBtn').onclick = () => {
        const dark = document.documentElement.dataset.theme === 'dark';
        document.documentElement.dataset.theme = dark ? '' : 'dark';
        localStorage.setItem('theme', dark ? 'light' : 'dark');
    };
    if (localStorage.getItem('theme') === 'dark' || (!localStorage.getItem('theme') && matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.documentElement.dataset.theme = 'dark';
    }
    byId('refreshBtn').onclick = () => refreshAll();
    byId('refreshJobs').onclick = () => refreshJobsOnly();
    byId('forecastSource').onchange = syncSourceFields;
    byId('layerSelect').onchange = event => selectLayer(event.target.value);
    byId('frameSlider').oninput = event => { stopPlayer(); setFrame(Number(event.target.value)); };
    byId('playBtn').onclick = () => {
        if (state.playerTimer) return stopPlayer();
        byId('playBtn').textContent = '❚❚';
        state.playerTimer = setInterval(() => setFrame((state.frameIndex + 1) % state.frames.length), 850);
    };
    byId('exportBtn').onclick = () => location.href = '/api/export/netcdf';
    byId('closeLog').onclick = () => byId('logDialog').close();

    byId('downloadForm').onsubmit = event => {
        event.preventDefault(); submitJob(event.target, '/api/task/download').catch(error => showToast(error.message, true));
    };
    byId('prepareForm').onsubmit = event => {
        event.preventDefault(); submitJob(event.target, '/api/task/prepare').catch(error => showToast(error.message, true));
    };
    byId('trainForm').onsubmit = async event => {
        event.preventDefault();
        try {
            const form = new FormData(event.target);
            document.querySelectorAll('.dataset-check:checked').forEach(element => form.append('dataset_dirs[]', element.value));
            if (!form.getAll('dataset_dirs[]').length) throw new Error('Выберите хотя бы один датасет');
            const data = await api('/api/task/train', {method: 'POST', body: form});
            showToast(`Обучение ${data.task_id.slice(0, 8)} поставлено в очередь`);
            await refreshAll(true); showPane('jobs');
        } catch (error) { showToast(error.message, true); }
    };
    byId('forecastForm').onsubmit = event => {
        event.preventDefault(); submitForecast(event.target).catch(error => showToast(error.message, true));
    };

    byId('downloadDate').valueAsDate = new Date();
    syncSourceFields(); refreshAll(true); loadStations();
    setInterval(refreshJobsOnly, 3000);
    setInterval(() => refreshAll(true), 30000);
}

document.addEventListener('DOMContentLoaded', init);
