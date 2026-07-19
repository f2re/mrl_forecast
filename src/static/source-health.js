(() => {
    const healthList = document.getElementById('sourceHealthList');
    const accessList = document.getElementById('sourceAccessList');
    const refreshButton = document.getElementById('sourceHealthRefresh');
    if (!healthList || !accessList) return;

    function text(element, value) {
        element.textContent = value == null ? '' : String(value);
        return element;
    }

    function badge(label, kind = '') {
        const element = document.createElement('span');
        element.className = `badge ${kind}`.trim();
        return text(element, label);
    }

    function statusKind(status) {
        if (status === 'available') return 'good';
        if (status === 'credential_required' || status === 'manual_registration') return 'warn';
        if (status === 'degraded') return 'info';
        return 'bad';
    }

    function empty(container, message) {
        container.replaceChildren(text(Object.assign(document.createElement('div'), {className: 'empty'}), message));
    }

    function renderHealth(payload) {
        const reports = Array.isArray(payload?.reports) ? payload.reports : [];
        if (!reports.length) {
            empty(healthList, 'Отчёт ещё не сформирован. Фоновая проверка запускается вместе с приложением.');
            return;
        }
        const fragment = document.createDocumentFragment();
        reports.forEach(report => {
            const row = document.createElement('div');
            row.className = 'list-item';
            const meta = document.createElement('div');
            meta.className = 'meta';
            meta.append(
                text(document.createElement('strong'), report.source_id),
                text(document.createElement('small'), report.message || 'Нет диагностического сообщения'),
            );
            const actions = document.createElement('div');
            actions.className = 'row-actions';
            actions.append(
                badge(report.status, statusKind(report.status)),
                badge(report.can_list ? 'список доступен' : 'нет списка', report.can_list ? 'good' : 'warn'),
                badge(report.can_download ? 'скачивание доступно' : 'скачивание недоступно', report.can_download ? 'good' : 'warn'),
            );
            row.append(meta, actions);
            fragment.append(row);
        });
        healthList.replaceChildren(fragment);
    }

    function renderAccess(sources) {
        const profiles = sources.filter(source =>
            source.access_mode !== 'open' || source.adapter_status !== 'active'
        );
        if (!profiles.length) {
            empty(accessList, 'Источники с регистрацией не обнаружены.');
            return;
        }
        const fragment = document.createDocumentFragment();
        profiles.forEach(source => {
            const row = document.createElement('div');
            row.className = 'list-item';
            const meta = document.createElement('div');
            meta.className = 'meta';
            meta.append(
                text(document.createElement('strong'), source.source_id),
                text(document.createElement('small'), `${source.access_mode} · ${source.native_format}`),
            );

            const details = document.createElement('details');
            const summary = text(document.createElement('summary'), 'Порядок доступа');
            details.append(summary);
            if (source.registration_url) {
                const link = document.createElement('a');
                link.href = source.registration_url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                text(link, 'Открыть страницу регистрации');
                details.append(link);
            }
            if (Array.isArray(source.registration_steps) && source.registration_steps.length) {
                const list = document.createElement('ol');
                source.registration_steps.forEach(step => {
                    list.append(text(document.createElement('li'), step));
                });
                details.append(list);
            }
            if (source.credential_env) {
                details.append(
                    text(
                        document.createElement('code'),
                        `python mrl.py sources --action configure --source ${source.source_id}`,
                    ),
                );
            }
            meta.append(details);

            const actions = document.createElement('div');
            actions.className = 'row-actions';
            actions.append(
                badge(source.credential_state || 'not_required', source.credential_state === 'present' ? 'good' : 'warn'),
                badge(source.download_supported ? 'адаптер скачивания' : 'ручной доступ', source.download_supported ? 'info' : 'warn'),
            );
            row.append(meta, actions);
            fragment.append(row);
        });
        accessList.replaceChildren(fragment);
    }

    async function load() {
        if (refreshButton) refreshButton.disabled = true;
        try {
            const [sourcesResponse, healthResponse] = await Promise.all([
                fetch('/api/sources', {cache: 'no-store'}),
                fetch(`/static/source_health.json?t=${Date.now()}`, {cache: 'no-store'}),
            ]);
            if (!sourcesResponse.ok) throw new Error(`Источники: HTTP ${sourcesResponse.status}`);
            renderAccess(await sourcesResponse.json());
            if (healthResponse.ok) {
                renderHealth(await healthResponse.json());
            } else if (healthResponse.status === 404) {
                empty(healthList, 'Фоновая проверка выполняется или приложение запущено без scripts/run_app.sh.');
            } else {
                throw new Error(`Health-report: HTTP ${healthResponse.status}`);
            }
        } catch (error) {
            empty(healthList, `Не удалось загрузить состояние источников: ${error.message}`);
        } finally {
            if (refreshButton) refreshButton.disabled = false;
        }
    }

    if (refreshButton) refreshButton.addEventListener('click', load);
    load();
    window.setInterval(load, 60000);
})();
