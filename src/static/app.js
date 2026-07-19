const state = {
    sources: [], raw: [], datasets: [], models: [], jobs: [], catalog: {},
    noaaStations: [], dwdStations: [], layers: {}, frames: [], labels: [], frame: 0, timer: null
};

const panes = {
    overview: ['Обзор', 'Состояние данных, моделей и вычислительных заданий'],
    forecast: ['Прогноз', 'Наблюдения и экспериментальный nowcast'],
    sources: ['Источники', 'Количественные, discovery и визуальные контуры'],
    data: ['Архив', 'Загрузка, каталог и canonical-предобработка'],
    training: ['Обучение', 'Выборка, балансировка и параметры'],
    jobs: ['Задания', 'Очередь и журналы локального worker'],
    models: ['Модели', 'Реестр, baseline и quality gate']
};

const layerTitles = {
    reflectivity: 'Поле отражаемости',
    motion: 'Перенос радиоэха',
    growth: 'Рост радиоэха',
    decay: 'Распад радиоэха',
    uncertainty: 'Неопределённость',
    valid_mask: 'Валидная область',
    coverage_mask: 'Геометрическое покрытие',
    clutter_mask: 'Исключённые помехи',
    interpolation_weight: 'Вес интерполяции'
};

const byId = id => document.getElementById(id);
const viewerImage = byId('viewerImage');
const viewerPlaceholder = byId('viewerPlaceholder');
const viewerTitle = byId('viewerTitle');
const frameSlider = byId('frameSlider');
const frameLabel = byId('frameLabel');
const playBtn = byId('playBtn');
const exportBtn = byId('exportBtn');
const layerSelect = byId('layerSelect');
const diagnostics = byId('diagnostics');
const forecastStation = byId('forecastStation');
const downloadSource = byId('downloadSource');
const downloadStation = byId('downloadStation');
const downloadStations = byId('downloadStations');
const prepareStep = byId('prepareStep');
const prepareLength = byId('prepareLength');
const logDialog = byId('logDialog');

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
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[character]));
}

function toast(message, error = false) {
    const element = byId('toast');
    element.textContent = message;
    element.className = `toast show${error ? ' error' : ''}`;
    clearTimeout(element.hideTimer);
    element.hideTimer = setTimeout(() => element.className = 'toast', 3200);
}

function statusBadge(status) {
    const className = status === 'completed' || status === 'published' ? 'good'
        : status === 'failed' || status === 'interrupted' || status === 'rejected_quality_gate' ? 'bad'
        : status === 'running' || status === 'queued' || status === 'cancelling' ? 'info' : 'warn';
    return `<span class="badge ${className}">${escapeHtml(status || 'unknown')}</span>`;
}

function fmtDate(value) {
    return value ? new Date(value).toLocaleString('ru-RU', {dateStyle: 'short', timeStyle: 'short'}) : '—';
}

function empty(text) {
    return `<div class="empty">${escapeHtml(text)}</div>`;
}

function showPane(name) {
    document.querySelectorAll('.pane').forEach(element => {
        element.classList.toggle('active', element.id === `pane-${name}`);
    });
    document.querySelectorAll('[data-pane]').forEach(element => {
        element.classList.toggle('active', element.dataset.pane === name);
    });
    byId('pageTitle').textContent = panes[name][0];
    byId('pageSubtitle').textContent = panes[name][1];
    window.scrollTo({top: 0, behavior: 'smooth'});
}

document.querySelectorAll('[data-pane]').forEach(element => {
    element.addEventListener('click', () => showPane(element.dataset.pane));
});
document.querySelectorAll('[data-go]').forEach(element => {
    element.addEventListener('click', () => showPane(element.dataset.go));
});

function handleAction(button) {
    const action = button.dataset.action;
    if (action === 'choose-archive') chooseArchive(button.dataset.path, button.dataset.source);
    else if (action === 'job-log') openLog(button.dataset.id);
    else if (action === 'cancel-job') cancelJob(button.dataset.id);
    else if (action === 'load-model') loadModel(button.dataset.path, button.dataset.name);
    else if (action === 'model-details') showModel(button.dataset.id);
}

document.addEventListener('click', event => {
    const button = event.target.closest('[data-action]');
    if (button) handleAction(button);
});

byId('themeBtn').onclick = () => {
    const dark = document.documentElement.dataset.theme === 'dark';
    document.documentElement.dataset.theme = dark ? '' : 'dark';
    localStorage.setItem('theme', dark ? 'light' : 'dark');
};
if (localStorage.getItem('theme') === 'dark' ||
    (!localStorage.getItem('theme') && matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.dataset.theme = 'dark';
}

function renderSources() {
    const html = state.sources.map(source => `
        <div class="card source-card"><div class="card-body">
            <div class="symbol">${source.training_allowed ? '◉' : source.visualization_allowed ? '◌' : '⌕'}</div>
            <h3>${escapeHtml(source.source_id)}</h3>
            <p>${escapeHtml(source.notes || source.native_format)}</p>
            <div class="badges">
                <span class="badge ${source.training_allowed ? 'good' : 'warn'}">${source.training_allowed ? 'обучение' : 'не для train'}</span>
                <span class="badge">${escapeHtml(source.native_format)}</span>
            </div>
        </div></div>`).join('');
    byId('sourcesGrid').innerHTML = html || empty('Источники не зарегистрированы');
    byId('overviewSources').innerHTML = state.sources.length ? state.sources.slice(0, 3).map(source => `
        <div class="card"><div class="card-body">
            <strong>${escapeHtml(source.source_id)}</strong>
            <p style="color:var(--muted);font-size:13px">${escapeHtml(source.native_format)}</p>
            ${source.training_allowed ? '<span class="badge good">quantitative</span>' : '<span class="badge warn">restricted</span>'}
        </div></div>`).join('') : empty('Нет данных');
}

function renderRaw() {
    const select = byId('prepareArchive');
    select.innerHTML = '<option value="">Выберите архив</option>' + state.raw
        .filter(item => item.status === 'completed')
        .map(item => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.folder_name)} · ${escapeHtml(item.source || '')}</option>`)
        .join('');
    byId('rawList').innerHTML = state.raw.length ? state.raw.map(item => `
        <div class="list-item">
            <div class="meta"><strong>${escapeHtml(item.folder_name)}</strong>
                <small>${escapeHtml(item.source || 'unknown')} · ${escapeHtml(item.station || '—')} · ${escapeHtml(item.date || '')} · ${item.downloaded_count || 0} файлов</small>
            </div>
            <div class="row-actions">${statusBadge(item.status)}
                <button class="button mini" data-action="choose-archive" data-path="${escapeHtml(item.path)}" data-source="${escapeHtml(item.source || '')}">Подготовить</button>
            </div>
        </div>`).join('') : empty('Архив пока пуст');
}

function chooseArchive(path, source) {
    byId('prepareArchive').value = path;
    const dwd = source === 'dwd-open-data';
    prepareStep.value = dwd ? 10 : 15;
    prepareLength.value = dwd ? 12 : 8;
    showPane('data');
    byId('prepareArchive').focus();
}

function classSummary(counts = {}) {
    return Object.entries(counts).filter(([, value]) => value)
        .map(([key, value]) => `${key}: ${value}`).join(' · ') || 'классы ещё не рассчитаны';
}

function renderDatasets() {
    const completed = state.datasets.filter(item => item.status === 'completed');
    byId('datasetPicker').innerHTML = completed.length ? completed.map(item => `
        <label class="dataset-option"><input type="checkbox" class="dataset-check" value="${escapeHtml(item.path)}">
            <span><strong>${escapeHtml(item.folder_name)}</strong>
                <small style="display:block;color:var(--muted)">${item.sample_count || 0} sequences · ${escapeHtml(item.pipeline?.pipeline_version || '')}<br>${escapeHtml(classSummary(item.class_counts))}</small>
            </span>
        </label>`).join('') : empty('Сначала подготовьте датасет');
    byId('datasetStats').innerHTML = state.datasets.length ? state.datasets.map(item => `
        <div class="card"><div class="card-body"><strong>${escapeHtml(item.folder_name)}</strong>
            <p style="color:var(--muted);font-size:13px">${escapeHtml(classSummary(item.class_counts))}</p>
            <span class="badge info">${item.sample_count || 0} samples</span>
        </div></div>`).join('') : empty('Нет статистики');
}

function renderModels() {
    byId('metricModels').textContent = state.models.length;
    byId('modelsGrid').innerHTML = state.models.length ? state.models.map(item => {
        const metrics = item.metrics || {};
        return `<div class="card"><div class="card-head"><h3>${escapeHtml(item.folder_name)}</h3>${statusBadge(item.status)}</div>
            <div class="card-body"><div class="badges">
                <span class="badge">${escapeHtml(item.model_architecture || 'unknown')}</span>
                <span class="badge info">${escapeHtml(item.pipeline_version || '')}</span>
                <span class="badge">${item.horizon_minutes || '—'} мин</span>
            </div>
            <p style="color:var(--muted);font-size:13px">Loss: ${metrics.best_val_loss != null ? Number(metrics.best_val_loss).toFixed(6) : '—'} · block: ${metrics.block_motion_mse != null ? Number(metrics.block_motion_mse).toFixed(6) : '—'} · ${escapeHtml(item.sampling || 'natural')}</p>
            <div class="row-actions">
                <button class="button mini" data-action="model-details" data-id="${escapeHtml(item.folder_name)}">Метаданные</button>
                <button class="button mini primary" data-action="load-model" data-path="${escapeHtml(item.path)}" data-name="${escapeHtml(item.folder_name)}" ${item.usable ? '' : 'disabled'}>Использовать</button>
            </div></div></div>`;
    }).join('') : empty('Реестр моделей пуст');
}

function jobHtml(job) {
    const cancellable = ['queued', 'running', 'cancelling'].includes(job.status);
    return `<div class="list-item job-line status-${escapeHtml(job.status)}">
        <div><span class="progress-dot"></span> <strong>${escapeHtml(job.kind)}</strong></div>
        <div class="meta"><small>${fmtDate(job.created_at)}${job.error ? ' · ' + escapeHtml(job.error) : ''}</small></div>
        <div class="row-actions">${statusBadge(job.status)}
            <button class="button mini" data-action="job-log" data-id="${job.id}">Журнал</button>
            ${cancellable ? `<button class="button mini danger" data-action="cancel-job" data-id="${job.id}">Отмена</button>` : ''}
        </div></div>`;
}

function renderJobs() {
    byId('jobsList').innerHTML = state.jobs.length ? state.jobs.map(jobHtml).join('') : empty('Заданий пока нет');
    byId('recentJobs').innerHTML = state.jobs.length ? state.jobs.slice(0, 5).map(jobHtml).join('') : empty('Заданий пока нет');
}

function renderCatalog() {
    byId('metricObservations').textContent = state.catalog.observations ?? 0;
    byId('catalogSummary').innerHTML = state.catalog.sources?.length ? state.catalog.sources.map(source => `
        <div class="list-item"><div class="meta"><strong>${escapeHtml(source.source)}</strong>
            <small>${source.stations} станций · ${source.observations} сроков · ${fmtDate(source.first_time_utc)} — ${fmtDate(source.last_time_utc)}</small>
        </div><span class="badge info">${source.observations}</span></div>`).join('') : empty('Каталог пуст. Выполните перестроение или загрузите архив.');
}

function renderMetrics() {
    byId('metricRaw').textContent = state.raw.length;
    byId('metricDatasets').textContent = state.datasets.length;
}

async function refreshAll(silent = false) {
    try {
        const [sources, raw, datasets, models, jobs, catalog] = await Promise.all([
            api('/api/sources'), api('/api/inventory/raw'), api('/api/inventory/datasets'),
            api('/api/inventory/models'), api('/api/jobs'), api('/api/catalog/summary')
        ]);
        Object.assign(state, {sources, raw, datasets, models, jobs, catalog});
        renderSources(); renderRaw(); renderDatasets(); renderModels(); renderJobs(); renderCatalog(); renderMetrics();
        if (!silent) toast('Данные обновлены');
    } catch (error) {
        toast(error.message, true);
    }
}

async function refreshJobsOnly() {
    try {
        state.jobs = await api('/api/jobs');
        renderJobs();
    } catch (_) {
        // Следующий цикл повторит запрос.
    }
}

byId('refreshBtn').onclick = () => refreshAll();
byId('refreshJobs').onclick = refreshJobsOnly;

async function submitJob(form, url) {
    const data = await api(url, {method: 'POST', body: new FormData(form)});
    toast(`Задание ${data.task_id.slice(0, 8)} поставлено в очередь`);
    await refreshAll(true);
    showPane('jobs');
}

byId('downloadForm').onsubmit = async event => {
    event.preventDefault();
    try { await submitJob(event.target, '/api/task/download'); }
    catch (error) { toast(error.message, true); }
};
byId('prepareForm').onsubmit = async event => {
    event.preventDefault();
    try { await submitJob(event.target, '/api/task/prepare'); }
    catch (error) { toast(error.message, true); }
};
byId('trainForm').onsubmit = async event => {
    event.preventDefault();
    try {
        const form = new FormData(event.target);
        document.querySelectorAll('.dataset-check:checked').forEach(element => form.append('dataset_dirs[]', element.value));
        if (!form.getAll('dataset_dirs[]').length) throw new Error('Выберите хотя бы один датасет');
        const data = await api('/api/task/train', {method: 'POST', body: form});
        toast(`Обучение ${data.task_id.slice(0, 8)} поставлено в очередь`);
        await refreshAll(true);
        showPane('jobs');
    } catch (error) { toast(error.message, true); }
};
byId('rebuildCatalogBtn').onclick = async () => {
    try {
        const data = await api('/api/task/catalog/rebuild', {method: 'POST'});
        toast(`Перестроение каталога ${data.task_id.slice(0, 8)} поставлено в очередь`);
        await refreshJobsOnly();
    } catch (error) { toast(error.message, true); }
};

async function openLog(id) {
    try {
        const data = await api(`/api/task/logs/${id}`);
        byId('logTitle').textContent = `${data.kind} · ${data.status}`;
        byId('logContent').textContent = data.logs || 'Журнал пока пуст';
        logDialog.showModal();
    } catch (error) { toast(error.message, true); }
}

async function cancelJob(id) {
    try {
        await api(`/api/jobs/${id}/cancel`, {method: 'POST'});
        toast('Запрошена отмена');
        refreshJobsOnly();
    } catch (error) { toast(error.message, true); }
}

async function loadModel(path, name) {
    try {
        const body = new FormData();
        body.append('model_path', path);
        const data = await api('/api/model/load', {method: 'POST', body});
        byId('activeModel').textContent = `${name} · ${data.model.model_architecture}`;
        toast('Модель загружена');
    } catch (error) { toast(error.message, true); }
}

async function showModel(id) {
    try {
        const data = await api(`/api/model/details/${encodeURIComponent(id)}`);
        byId('logTitle').textContent = `Модель ${id}`;
        byId('logContent').textContent = JSON.stringify(data.metadata, null, 2);
        logDialog.showModal();
    } catch (error) { toast(error.message, true); }
}

const forecastSource = byId('forecastSource');
function syncForecastFields() {
    const value = forecastSource.value;
    byId('localField').hidden = value !== 'local';
    byId('uploadField').hidden = value !== 'upload';
    byId('forecastStationField').hidden = !['aws', 'ftp'].includes(value);
}
forecastSource.onchange = syncForecastFields;
syncForecastFields();

function stopPlayer() {
    clearInterval(state.timer);
    state.timer = null;
    playBtn.textContent = '▶';
}

function setFrame(index) {
    if (!state.frames.length) return;
    state.frame = Math.max(0, Math.min(index, state.frames.length - 1));
    viewerImage.src = state.frames[state.frame];
    viewerImage.style.display = 'block';
    viewerPlaceholder.style.display = 'none';
    frameSlider.value = state.frame;
    frameLabel.textContent = state.labels[state.frame];
}

function selectLayer(name) {
    stopPlayer();
    const items = name === 'reflectivity'
        ? [...(state.layers.history || []), ...(state.layers.reflectivity || [])]
        : (state.layers[name] || []);
    state.frames = items.map(item => `data:image/png;base64,${item.data}`);
    state.labels = items.map(item => item.label);
    viewerTitle.textContent = layerTitles[name] || name;
    layerSelect.value = name;
    frameSlider.max = Math.max(state.frames.length - 1, 0);
    frameSlider.disabled = !state.frames.length;
    playBtn.disabled = !state.frames.length;
    if (state.frames.length) setFrame(name === 'reflectivity' ? Math.max((state.layers.history || []).length - 1, 0) : 0);
}

function populateLayers() {
    const names = [
        'reflectivity', 'motion', 'growth', 'decay', 'uncertainty',
        'valid_mask', 'coverage_mask', 'clutter_mask', 'interpolation_weight'
    ].filter(name => (state.layers[name] || []).length);
    layerSelect.innerHTML = names.map(name => `<option value="${name}">${escapeHtml(layerTitles[name] || name)}</option>`).join('');
    layerSelect.disabled = !names.length;
    selectLayer(names.includes('reflectivity') ? 'reflectivity' : names[0]);
}

layerSelect.onchange = event => selectLayer(event.target.value);
playBtn.onclick = () => {
    if (state.timer) return stopPlayer();
    playBtn.textContent = '❚❚';
    state.timer = setInterval(() => setFrame((state.frame + 1) % state.frames.length), 850);
};
frameSlider.oninput = event => { stopPlayer(); setFrame(Number(event.target.value)); };
exportBtn.onclick = () => location.href = '/api/export/netcdf';

byId('forecastForm').onsubmit = async event => {
    event.preventDefault();
    stopPlayer();
    try {
        const data = await api('/api/predict', {method: 'POST', body: new FormData(event.target)});
        state.layers = {history: data.history || [], ...(data.layers || {reflectivity: data.forecast || []})};
        if (!state.layers.reflectivity) state.layers.reflectivity = data.forecast || [];
        populateLayers();
        exportBtn.disabled = false;
        const evolution = data.evolution_diagnostics || {};
        const quality = data.quality_diagnostics || {};
        diagnostics.innerHTML = `
            <span class="badge info">${escapeHtml(data.model_architecture)}</span>
            <span class="badge">${escapeHtml(data.pipeline_version)}</span>
            <span class="badge">${data.grid?.width || '—'}×${data.grid?.height || '—'}</span>
            <span class="badge">max ${Number(data.diagnostics.max_dbz).toFixed(1)} dBZ</span>
            ${evolution.mean_motion_pixels != null ? `<span class="badge">motion ${Number(evolution.mean_motion_pixels).toFixed(2)} px</span>` : ''}
            ${quality.valid_fraction != null ? `<span class="badge">valid ${(100 * Number(quality.valid_fraction)).toFixed(1)}%</span>` : ''}
            ${quality.clutter_fraction != null && Number(quality.clutter_fraction) > 0 ? `<span class="badge warn">clutter ${(100 * Number(quality.clutter_fraction)).toFixed(2)}%</span>` : ''}
            <span class="badge warn">экспериментальный продукт</span>`;
    } catch (error) { toast(error.message, true); }
};

async function loadNoaaStations() {
    try {
        state.noaaStations = await api('/api/ftp/stations');
        forecastStation.innerHTML = state.noaaStations.map(station => `
            <option value="${escapeHtml(station.code)}">${escapeHtml(station.code.toUpperCase())} · ${escapeHtml(station.name)}</option>`).join('');
    } catch (_) {
        state.noaaStations = [{code: 'kokx', name: 'KOKX'}];
    }
}

async function loadDwdStations() {
    try { state.dwdStations = await api('/api/dwd/stations'); }
    catch (_) { state.dwdStations = ['ess']; }
}

function syncDownloadSource() {
    const dwd = downloadSource.value === 'dwd-open-data';
    const values = dwd
        ? state.dwdStations.map(code => ({value: code, label: code.toUpperCase()}))
        : state.noaaStations.map(station => ({value: station.code.toUpperCase(), label: `${station.code.toUpperCase()} · ${station.name}`}));
    downloadStations.innerHTML = values.map(item => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`).join('');
    downloadStation.value = values[0]?.value || (dwd ? 'ess' : 'KOKX');
    downloadStation.maxLength = dwd ? 3 : 4;
}

downloadSource.onchange = syncDownloadSource;
byId('downloadDate').valueAsDate = new Date();

Promise.all([loadNoaaStations(), loadDwdStations()]).then(syncDownloadSource);
refreshAll(true);
setInterval(refreshJobsOnly, 3000);
setInterval(() => refreshAll(true), 30000);
