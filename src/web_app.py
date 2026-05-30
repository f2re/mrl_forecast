print("DEBUG: Starting imports...")
import matplotlib
matplotlib.use('Agg')
from map_visualization import generate_sequence_plots
import io
import os
import pathlib
import base64
import sys
import subprocess
import threading
from typing import List, Optional, Tuple, Dict

print("DEBUG: Importing numpy...")
import numpy as np
print("DEBUG: Importing flask...")
from flask import Flask, request, render_template, redirect, url_for, jsonify

print("DEBUG: Importing torch...")
import torch
import torch.nn as nn

# Добавляем src в путь
sys.path.append(str(pathlib.Path(__file__).parent))

print("DEBUG: Importing local modules...")
print("DEBUG: Importing ConvLSTM...")
from train_nowcasting_model import ConvLSTM
print("DEBUG: Importing adapters...")
from adapters import LocalDirectoryAdapter, RainViewerAdapter, NOAAFTPAdapter
print("DEBUG: Importing generate_sequence_plots...")
from map_visualization import generate_sequence_plots
print("DEBUG: Importing metadata_utils...")
from metadata_utils import scan_inventory, load_metadata
print("DEBUG: Importing export_utils...")
from export_utils import save_forecast_to_netcdf
import datetime

print("DEBUG: Imports finished.")


app = Flask(__name__, template_folder='../templates')
app.config['RAW_DATA_DIR'] = 'data/raw/archive'
app.config['DATASETS_DIR'] = 'data/processed_archive'
app.config['MODELS_REGISTRY_DIR'] = 'models/registry'
app.config['LOCAL_DATA_DIR'] = os.environ.get('RADAR_DATA_DIR', 'data/processed')

# Глобальное хранилище для фоновых задач
class TaskRunner:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}

    def run(self, task_id: str, command: List[str]):
        if task_id in self.tasks and self.tasks[task_id]['process'].poll() is None:
            return False, "Задача уже запущена"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.tasks[task_id] = {
            'process': process,
            'logs': [],
            'status': 'running'
        }

        def monitor():
            for line in process.stdout:
                self.tasks[task_id]['logs'].append(line)
                if len(self.tasks[task_id]['logs']) > 500: # Храним последние 500 строк
                    self.tasks[task_id]['logs'].pop(0)
            
            process.wait()
            self.tasks[task_id]['status'] = 'finished' if process.returncode == 0 else 'failed'

        threading.Thread(target=monitor, daemon=True).start()
        return True, "Задача запущена"

    def get_logs(self, task_id: str):
        if task_id not in self.tasks:
            return None
        return {
            'logs': "".join(self.tasks[task_id]['logs']),
            'status': self.tasks[task_id]['status']
        }

task_runner = TaskRunner()

# Configuration
CHECKPOINT_PATH = os.environ.get('NOWCAST_MODEL_CHECKPOINT', 'models/checkpoints/best_model.pt')

# Global model and settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: Optional[nn.Module] = None
INPUT_LENGTH: int = 4
TARGET_LENGTH: int = 4
HIDDEN_CHANNELS: List[int] = []


def _load_model(checkpoint_path: str):
    """Load a ConvLSTM model checkpoint and update global settings."""
    global model, INPUT_LENGTH, TARGET_LENGTH, HIDDEN_CHANNELS
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    INPUT_LENGTH = checkpoint.get('input_length', 4)
    TARGET_LENGTH = checkpoint.get('target_length', 4)
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


def _preprocess_input(array: np.ndarray) -> torch.Tensor:
    """Prepare a radar sequence for model inference."""
    # Ensure (T, H, W)
    if array.ndim == 4 and array.shape[3] == 1: # (T, H, W, 1)
        array = array.squeeze(-1)
    
    MAX_DBZ = 70.0
    array = np.clip(array, 0.0, MAX_DBZ)
    array = array / MAX_DBZ
    
    if array.shape[0] > INPUT_LENGTH:
        array = array[-INPUT_LENGTH:]
    elif array.shape[0] < INPUT_LENGTH:
        pad_shape = (INPUT_LENGTH - array.shape[0],) + array.shape[1:]
        pad = np.zeros(pad_shape, dtype=array.dtype)
        array = np.concatenate([pad, array], axis=0)
        
    tensor = torch.from_numpy(array).unsqueeze(1).unsqueeze(0).float().to(device)
    return tensor


@app.route('/')
def index():
    return render_template('index.html', local_dir=app.config['LOCAL_DATA_DIR'])


@app.route('/api/ftp/stations', methods=['GET'])
def get_ftp_stations():
    adapter = NOAAFTPAdapter()
    stations = adapter.get_available_stations()
    return jsonify(stations)


@app.route('/api/ftp/times', methods=['GET'])
def get_ftp_times():
    station = request.args.get('station', 'kokx')
    adapter = NOAAFTPAdapter()
    times = adapter.get_available_times(station)
    return jsonify(times)


@app.route('/api/predict', methods=['POST'])
def predict():
    global LAST_FORECAST
    source_type = request.form.get('source_type')
    station_code = request.form.get('ftp_station', 'kokx')
    status_msg = "Успешно"
    
    try:
        if source_type == 'upload':
            uploaded = request.files.get('file')
            if not uploaded:
                return jsonify({'error': 'Файл не выбран'}), 400
            data = np.load(uploaded)
            array = data['arr_0'] if isinstance(data, np.lib.npyio.NpzFile) else data
            status_msg = "Файл загружен вручную"
        
        elif source_type == 'local':
            path = request.form.get('local_path', app.config['LOCAL_DATA_DIR'])
            adapter = LocalDirectoryAdapter(path)
            array, status_msg = adapter.get_latest_sequence(INPUT_LENGTH)
        
        elif source_type == 'ftp':
            time_id = request.form.get('ftp_time', 'latest')
            adapter = NOAAFTPAdapter()
            array, status_msg = adapter.get_latest_sequence(INPUT_LENGTH, station_code=station_code, end_file_id=time_id)
        
        else:
            return jsonify({'error': 'Неверный тип источника'}), 400

        if model is None:
            return jsonify({'error': 'Модель ИИ не загружена (чекпоинт не найден)'}), 500

        # Preprocess and predict
        tensor_input = _preprocess_input(array)
        with torch.no_grad():
            preds, _ = model(tensor_input)
        
        in_data = tensor_input.cpu().squeeze(0).squeeze(1).numpy()
        pred_data = preds.cpu().squeeze(0).squeeze(1).numpy()
        
        png_list = generate_sequence_plots(in_data, pred_data, INPUT_LENGTH, station_code=station_code)
        
        # Prepare JSON response
        history = []
        for idx in range(INPUT_LENGTH):
            b64 = base64.b64encode(png_list[idx]).decode('utf-8')
            label = f"T{(INPUT_LENGTH - idx - 1) * -15 if idx < INPUT_LENGTH-1 else '-0'} мин"
            history.append({'data': b64, 'label': label})
            
        forecast = []
        for idx in range(TARGET_LENGTH):
            b64 = base64.b64encode(png_list[INPUT_LENGTH + idx]).decode('utf-8')
            label = f"T+{(idx + 1) * 15} мин"
            forecast.append({'data': b64, 'label': label})
            
        # Сохраняем для экспорта
        LAST_FORECAST = {
            'data': pred_data * 70.0, # De-normalize back to dBZ
            'base_time': datetime.datetime.now(), # В реальности брать время из радара
            'station': request.form.get('ftp_station', 'unknown')
        }
            
        return jsonify({
            'history': history,
            'forecast': forecast,
            'status': status_msg
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
    return jsonify(scan_inventory(app.config['MODELS_REGISTRY_DIR']))


@app.route('/api/task/download', methods=['POST'])
def start_download():
    station = request.form.get('station', 'KOKX')
    date = request.form.get('date', '')
    count = request.form.get('count', '50')
    
    cmd = ['bash', 'scripts/download.sh', station, date, count]
    success, msg = task_runner.run('download', cmd)
    return jsonify({'success': success, 'message': msg})


@app.route('/api/task/prepare', methods=['POST'])
def start_prepare():
    archive_dir = request.form.get('archive_dir')
    seq_len = request.form.get('seq_len', '8')
    
    if not archive_dir:
        return jsonify({'success': False, 'message': 'Директория архива не указана'})
    
    cmd = ['bash', 'scripts/prepare.sh', seq_len, archive_dir]
    success, msg = task_runner.run('prepare', cmd)
    return jsonify({'success': success, 'message': msg})

LAST_FORECAST = {}

@app.route('/api/task/train', methods=['POST'])
def start_train():
    dataset_dirs = request.form.getlist('dataset_dirs[]')
    epochs = request.form.get('epochs', '10')
    batch_size = request.form.get('batch_size', '4')
    lr = request.form.get('lr', '1e-4')
    val_split = request.form.get('val_split', '0.2')
    lead_time = request.form.get('lead_time', '4')

    if not dataset_dirs:
        return jsonify({'success': False, 'message': 'Датасеты не выбраны'})

    dirs_str = ",".join(dataset_dirs)
    cmd = ['bash', 'scripts/train.sh', epochs, batch_size, lr, dirs_str, val_split, lead_time]
    success, msg = task_runner.run('train', cmd)
    return jsonify({'success': success, 'message': msg})


@app.route('/api/model/details/<model_id>', methods=['GET'])
def get_model_details(model_id):
    model_path = os.path.join(app.config['MODELS_REGISTRY_DIR'], model_id)
    meta = load_metadata(model_path)
    if not meta:
        return jsonify({'error': 'Метаданные не найдены'}), 404

    # Проверяем наличие графика
    plot_path = os.path.join(model_path, 'learning_curve.png')
    plot_b64 = None
    if os.path.exists(plot_path):
        with open(plot_path, "rb") as img_file:
            plot_b64 = base64.b64encode(img_file.read()).decode('utf-8')

    return jsonify({
        'metadata': meta,
        'plot': plot_b64
    })


@app.route('/api/export/netcdf', methods=['GET'])
def export_netcdf():
    if not LAST_FORECAST or 'data' not in LAST_FORECAST:
        return jsonify({'error': 'Прогноз не найден. Сначала запустите инференс.'}), 404

    try:
        export_dir = 'data/exports'
        os.makedirs(export_dir, exist_ok=True)

        station = LAST_FORECAST.get('station', 'unknown')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"forecast_{station}_{timestamp}.nc"
        file_path = os.path.join(export_dir, filename)

        save_forecast_to_netcdf(
            forecast_data=LAST_FORECAST['data'],
            base_time=LAST_FORECAST['base_time'],
            station_id=station,
            output_path=file_path
        )

        from flask import send_file
        return send_file(os.path.abspath(file_path), as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Ошибка экспорта: {str(e)}'}), 500


@app.route('/api/task/logs/<task_id>', methods=['GET'])
def get_task_logs(task_id):
    result = task_runner.get_logs(task_id)
    if not result:
        return jsonify({'error': 'Задача не найдена'}), 404
    return jsonify(result)


@app.route('/api/data/preview', methods=['GET'])
def preview_data():
    station = request.args.get('station', 'KOKX').upper()
    date_str = request.args.get('date')

    if not date_str:
        return jsonify({'error': 'Дата не указана'}), 400

    try:
        # Attempt to import the dependency and provide guidance if missing
        try:
            import nexradaws  # type: ignore
        except ImportError:
            return jsonify({
                'error': 'Библиотека nexradaws не установлена. Установите её командой `pip install nexradaws`.'
            }), 500

        # Attempt to create the AWS interface; catch initialization errors
        try:
            conn = nexradaws.NexradAwsInterface()
        except Exception as init_exc:
            return jsonify({
                'error': f'Ошибка инициализации NexradAwsInterface: {init_exc}'
            }), 500

        # Validate the date format
        try:
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Неверный формат даты. Используйте YYYY‑MM‑DD.'}), 400

        # Fetch scans and handle API errors
        try:
            scans = conn.get_avail_scans(dt.year, dt.month, dt.day, station)
        except Exception as fetch_exc:
            return jsonify({'error': f'Не удалось получить список сканов: {fetch_exc}'}), 500

        # Group scans by hour
        availability = [0] * 24
        for scan in scans:
            try:
                availability[scan.scan_time.hour] += 1
            except AttributeError:
                continue

        return jsonify({
            'station': station,
            'date': date_str,
            'total_scans': len(scans),
            'hourly_counts': availability,
            'has_data': len(scans) > 0
        })
    except Exception as e:
        # Generic fallback for unexpected errors
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    _load_model(CHECKPOINT_PATH)
    port = int(os.environ.get('PORT', 5005))
    app.run(host='0.0.0.0', port=port, debug=True)
