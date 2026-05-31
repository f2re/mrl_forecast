import matplotlib
matplotlib.use('Agg')

import base64
import datetime
import os
import pathlib
import sys
from typing import Dict, List, Optional

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

import numpy as np
from flask import Flask, jsonify, render_template, request

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).parent))

from adapters import DemoRadarAdapter, LocalDirectoryAdapter, NOAAAWSAdapter, NOAAFTPAdapter
from config import FORECAST_STEP_MINUTES, MAX_DBZ, PRODUCT_NAME
from export_utils import save_forecast_to_netcdf
from forecast_quality import summarize_forecast
from map_visualization import RADAR_COORDS, generate_sequence_plots
from metadata_utils import load_metadata, scan_inventory
from radar_pipeline import PIPELINE_VERSION
from train_nowcasting_model import ConvLSTM


app = Flask(__name__, template_folder='../templates')
app.config['RAW_DATA_DIR'] = 'data/raw/archive'
app.config['DATASETS_DIR'] = 'data/processed_archive'
app.config['MODELS_REGISTRY_DIR'] = 'models/registry'
app.config['LOCAL_DATA_DIR'] = os.environ.get('RADAR_DATA_DIR', 'data/processed')

CHECKPOINT_PATH = os.environ.get('NOWCAST_MODEL_CHECKPOINT', 'models/checkpoints/best_model.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: Optional[nn.Module] = None
INPUT_LENGTH: int = 4
TARGET_LENGTH: int = 4
HIDDEN_CHANNELS: List[int] = []
CURRENT_MODEL_INFO: Dict = {}
LAST_FORECAST = {}


def _lead_times_minutes(target_length: int) -> List[int]:
    return [FORECAST_STEP_MINUTES * (idx + 1) for idx in range(target_length)]


def _confidence_by_lead(lead_times: List[int]) -> List[str]:
    values = []
    for lead in lead_times:
        if lead <= 60:
            values.append('normal_experimental')
        elif lead <= 120:
            values.append('reduced')
        elif lead <= 180:
            values.append('experimental_low_confidence')
        else:
            values.append('unsupported')
    return values


def is_model_usable(model_path: str) -> bool:
    """Return whether a registry model is complete and has a checkpoint."""
    metadata = load_metadata(model_path)
    return bool(
        metadata
        and metadata.get('status') in ('completed', 'published')
        and os.path.exists(os.path.join(model_path, 'best_model.pt'))
    )


def _load_model(checkpoint_path: str):
    """Load a ConvLSTM model checkpoint and update global settings."""
    global model, INPUT_LENGTH, TARGET_LENGTH, HIDDEN_CHANNELS, CURRENT_MODEL_INFO

    if os.path.isdir(checkpoint_path):
        metadata = load_metadata(checkpoint_path)
        if metadata and not is_model_usable(checkpoint_path):
            raise ValueError(
                f"Модель {os.path.basename(checkpoint_path)} не прошла quality gate: "
                f"{metadata.get('status', 'unknown')}"
            )
        candidate = os.path.join(checkpoint_path, 'best_model.pt')
        if os.path.exists(candidate):
            checkpoint_path = candidate
        else:
            raise FileNotFoundError(f"В директории {checkpoint_path} не найден файл best_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_pipeline = checkpoint.get('pipeline_version')
    if checkpoint_pipeline and checkpoint_pipeline != PIPELINE_VERSION:
        raise ValueError(
            f"Модель использует несовместимый pipeline {checkpoint_pipeline}; ожидается {PIPELINE_VERSION}"
        )
    hyperparameters = checkpoint.get('hyperparameters', {})
    INPUT_LENGTH = checkpoint.get('input_length', hyperparameters.get('input_length', 4))
    TARGET_LENGTH = checkpoint.get('target_length', hyperparameters.get('target_length', 4))
    HIDDEN_CHANNELS = checkpoint.get('hidden_channels', [32, 1])
    mdl = ConvLSTM(
        input_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        kernel_size=(3, 3),
        output_steps=TARGET_LENGTH,
    )
    mdl.load_state_dict(checkpoint['model_state_dict'])
    mdl.to(device)
    mdl.eval()
    model = mdl
    CURRENT_MODEL_INFO = {
        'path': checkpoint_path,
        'model_id': os.path.basename(os.path.dirname(checkpoint_path)),
        'pipeline_version': checkpoint_pipeline or 'legacy',
        'model_architecture': checkpoint.get('model_architecture', 'convlstm_baseline'),
    }


def _preprocess_input(array: np.ndarray) -> torch.Tensor:
    """Prepare a radar sequence for model inference."""
    if array.ndim == 4 and array.shape[3] == 1:
        array = array.squeeze(-1)
    array = np.clip(array, 0.0, MAX_DBZ)
    array = array / MAX_DBZ
    if array.shape[0] > INPUT_LENGTH:
        array = array[-INPUT_LENGTH:]
    elif array.shape[0] < INPUT_LENGTH:
        pad_shape = (INPUT_LENGTH - array.shape[0],) + array.shape[1:]
        pad = np.zeros(pad_shape, dtype=array.dtype)
        array = np.concatenate([pad, array], axis=0)
    return torch.from_numpy(array).unsqueeze(1).unsqueeze(0).float().to(device)


@app.route('/')
def index():
    return render_template('index.html', local_dir=app.config['LOCAL_DATA_DIR'])


@app.route('/api/ftp/stations', methods=['GET'])
def get_ftp_stations():
    return jsonify(NOAAFTPAdapter().get_available_stations())


@app.route('/api/ftp/times', methods=['GET'])
def get_ftp_times():
    station = request.args.get('station', 'kokx')
    return jsonify(NOAAFTPAdapter().get_available_times(station))


@app.route('/api/predict', methods=['POST'])
def predict():
    global LAST_FORECAST
    source_type = request.form.get('source_type')
    station_code = request.form.get('ftp_station', 'kokx')
    status_msg = 'Успешно'
    sequence = None

    try:
        if source_type == 'upload':
            uploaded = request.files.get('file')
            if not uploaded:
                return jsonify({'error': 'Файл не выбран'}), 400
            data = np.load(uploaded)
            array = data['arr_0'] if isinstance(data, np.lib.npyio.NpzFile) else data
            now = datetime.datetime.now(datetime.UTC)
            timestamps = [
                now - datetime.timedelta(minutes=(array.shape[0] - idx - 1) * FORECAST_STEP_MINUTES)
                for idx in range(array.shape[0])
            ]
            status_msg = 'Файл загружен вручную'
        elif source_type == 'local':
            sequence = LocalDirectoryAdapter(request.form.get('local_path', app.config['LOCAL_DATA_DIR'])).get_latest_sequence(INPUT_LENGTH)
            array, timestamps, status_msg = sequence
        elif source_type == 'ftp':
            sequence = NOAAFTPAdapter().get_latest_sequence(
                INPUT_LENGTH,
                station_code=station_code,
                end_file_id=request.form.get('ftp_time', 'latest'),
            )
            array, timestamps, status_msg = sequence
        elif source_type == 'aws':
            sequence = NOAAAWSAdapter().get_latest_sequence(INPUT_LENGTH, station_code=station_code)
            array, timestamps, status_msg = sequence
        elif source_type == 'demo':
            sequence = DemoRadarAdapter().get_latest_sequence(INPUT_LENGTH)
            array, timestamps, status_msg = sequence
        else:
            return jsonify({'error': 'Неверный тип источника'}), 400

        if model is None:
            return jsonify({'error': 'Модель ИИ не загружена (чекпоинт не найден)'}), 500

        tensor_input = _preprocess_input(array)
        with torch.no_grad():
            preds, _ = model(tensor_input)

        in_data = tensor_input.cpu().squeeze(0).squeeze(1).numpy()
        pred_data = preds.cpu().squeeze(0).squeeze(1).numpy()
        diagnostics = summarize_forecast(pred_data * MAX_DBZ)
        source_status = sequence.status if sequence is not None else 'observed'
        if diagnostics['uniform_field_anomaly']:
            return jsonify({
                'error': 'Прогноз отклонен: обнаружен почти однородный слой отражаемости.',
                'source_status': source_status,
                'diagnostics': diagnostics,
            }), 422

        last_ts = timestamps[-1] if timestamps else datetime.datetime.now(datetime.UTC)
        png_list = generate_sequence_plots(
            in_data,
            pred_data,
            INPUT_LENGTH,
            station_code=station_code,
            start_datetime=last_ts,
            history_timestamps=timestamps,
            interval_minutes=FORECAST_STEP_MINUTES,
        )

        history = []
        for idx in range(INPUT_LENGTH):
            b64 = base64.b64encode(png_list[idx]).decode('utf-8')
            ts = timestamps[idx] if idx < len(timestamps) else last_ts
            history.append({'data': b64, 'label': ts.strftime('%H:%M') + ' UTC'})

        lead_times = _lead_times_minutes(TARGET_LENGTH)
        forecast = []
        for idx, lead_time in enumerate(lead_times):
            b64 = base64.b64encode(png_list[INPUT_LENGTH + idx]).decode('utf-8')
            ts = last_ts + datetime.timedelta(minutes=lead_time)
            forecast.append({
                'data': b64,
                'label': f'{ts:%H:%M} UTC (T+{lead_time} мин)',
                'lead_time_minutes': lead_time,
            })

        horizon_minutes = lead_times[-1] if lead_times else 0
        LAST_FORECAST = {
            'data': pred_data * MAX_DBZ,
            'base_time': last_ts,
            'station': request.form.get('ftp_station', 'unknown'),
            'source': source_type,
            'model_id': CURRENT_MODEL_INFO.get('model_id', 'unknown'),
            'model_architecture': CURRENT_MODEL_INFO.get('model_architecture', 'unknown'),
            'pipeline_version': CURRENT_MODEL_INFO.get('pipeline_version', 'legacy'),
            'forecast_step_minutes': FORECAST_STEP_MINUTES,
        }

        return jsonify({
            'product': PRODUCT_NAME,
            'units': 'dBZ',
            'base_time_utc': last_ts.isoformat(),
            'forecast_step_minutes': FORECAST_STEP_MINUTES,
            'horizon_minutes': horizon_minutes,
            'lead_times_minutes': lead_times,
            'pipeline_version': CURRENT_MODEL_INFO.get('pipeline_version', 'legacy'),
            'model_id': CURRENT_MODEL_INFO.get('model_id', 'unknown'),
            'model_architecture': CURRENT_MODEL_INFO.get('model_architecture', 'unknown'),
            'history': history,
            'forecast': forecast,
            'status': status_msg,
            'source_status': source_status,
            'confidence_by_lead': _confidence_by_lead(lead_times),
            'diagnostics': diagnostics,
            'warnings': ['not_official_warning', 'reflectivity_only_no_nwp'],
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model_route():
    model_path = request.form.get('model_path')
    if not model_path:
        return jsonify({'error': 'Путь к модели не указан'}), 400
    try:
        _load_model(model_path)
        return jsonify({'success': True, 'message': f'Модель {os.path.basename(model_path)} успешно загружена'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inventory/raw', methods=['GET'])
def get_raw_inventory():
    return jsonify(scan_inventory(app.config['RAW_DATA_DIR']))


@app.route('/api/inventory/datasets', methods=['GET'])
def get_datasets_inventory():
    return jsonify(scan_inventory(app.config['DATASETS_DIR']))


@app.route('/api/inventory/models', methods=['GET'])
def get_models_inventory():
    items = scan_inventory(app.config['MODELS_REGISTRY_DIR'])
    for item in items:
        item['usable'] = is_model_usable(item['path'])
    return jsonify(items)


@app.route('/api/task/download', methods=['POST'])
@app.route('/api/task/prepare', methods=['POST'])
@app.route('/api/task/train', methods=['POST'])
def background_task_disabled():
    return jsonify({
        'success': False,
        'message': 'Фоновые задачи отключены до внедрения безопасного job runner. Используйте scripts/*.sh вручную.',
    }), 503


@app.route('/api/model/details/<model_id>', methods=['GET'])
def get_model_details(model_id):
    model_path = os.path.join(app.config['MODELS_REGISTRY_DIR'], model_id)
    meta = load_metadata(model_path)
    if not meta:
        return jsonify({'error': 'Метаданные не найдены'}), 404
    plot_path = os.path.join(model_path, 'learning_curve.png')
    plot_b64 = None
    if os.path.exists(plot_path):
        with open(plot_path, 'rb') as img_file:
            plot_b64 = base64.b64encode(img_file.read()).decode('utf-8')
    return jsonify({'metadata': meta, 'plot': plot_b64})


@app.route('/api/export/netcdf', methods=['GET'])
def export_netcdf():
    if not LAST_FORECAST or 'data' not in LAST_FORECAST:
        return jsonify({'error': 'Прогноз не найден. Сначала запустите инференс.'}), 404
    try:
        export_dir = 'data/exports'
        os.makedirs(export_dir, exist_ok=True)
        station = LAST_FORECAST.get('station', 'unknown')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(export_dir, f'forecast_{station}_{timestamp}.nc')
        save_forecast_to_netcdf(
            forecast_data=LAST_FORECAST['data'],
            base_time=LAST_FORECAST['base_time'],
            station_id=station,
            output_path=file_path,
            station_lon=RADAR_COORDS.get(station.lower(), (None, None))[0],
            station_lat=RADAR_COORDS.get(station.lower(), (None, None))[1],
            model_id=LAST_FORECAST.get('model_id', 'unknown'),
            model_architecture=LAST_FORECAST.get('model_architecture', 'unknown'),
            source=LAST_FORECAST.get('source', 'unknown'),
            pipeline_version=LAST_FORECAST.get('pipeline_version', 'legacy'),
            interval_minutes=LAST_FORECAST.get('forecast_step_minutes', FORECAST_STEP_MINUTES),
        )
        from flask import send_file
        return send_file(os.path.abspath(file_path), as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Ошибка экспорта: {str(e)}'}), 500


@app.route('/api/task/logs/<task_id>', methods=['GET'])
def get_task_logs(task_id):
    return jsonify({'error': 'Фоновый task runner отключен'}), 503


@app.route('/api/data/preview', methods=['GET'])
def preview_data():
    station = request.args.get('station', 'KOKX').upper()
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Дата не указана'}), 400
    try:
        import nexradaws  # type: ignore
        conn = nexradaws.NexradAwsInterface()
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        scans = conn.get_avail_scans(dt.year, dt.month, dt.day, station)
        availability = [0] * 24
        for scan in scans:
            if hasattr(scan, 'scan_time'):
                availability[scan.scan_time.hour] += 1
        return jsonify({
            'station': station,
            'date': date_str,
            'total_scans': len(scans),
            'hourly_counts': availability,
            'has_data': len(scans) > 0,
        })
    except ValueError:
        return jsonify({'error': 'Неверный формат даты. Используйте YYYY‑MM‑DD.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    _load_model(CHECKPOINT_PATH)
    port = int(os.environ.get('PORT', 5005))
    app.run(host='0.0.0.0', port=port, debug=True)
