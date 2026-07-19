import matplotlib
matplotlib.use("Agg")

import base64
import datetime
import os
import pathlib
import sys
from typing import Dict, List, Optional

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

import numpy as np
from flask import Flask, jsonify, render_template, request
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).parent))

from adapters import DemoRadarAdapter, LocalDirectoryAdapter, NOAAAWSAdapter, NOAAFTPAdapter
from config import FORECAST_STEP_MINUTES, MAX_DBZ, PRODUCT_NAME
from convlstm import ConvLSTM
from export_utils import save_forecast_to_netcdf
from forecast_quality import summarize_forecast
from map_visualization import RADAR_COORDS, generate_sequence_plots
from metadata_utils import load_metadata, scan_inventory
from radar_pipeline import CANONICAL_PIPELINE_VERSION, PIPELINE_VERSION, RadarPipeline
from source_registry import build_default_source_registry

app = Flask(__name__, template_folder="../templates")
app.config["RAW_DATA_DIR"] = "data/raw/archive"
app.config["DATASETS_DIR"] = "data/processed_archive"
app.config["MODELS_REGISTRY_DIR"] = "models/registry"
app.config["LOCAL_DATA_DIR"] = os.environ.get("RADAR_DATA_DIR", "data/processed")

CHECKPOINT_PATH = os.environ.get("NOWCAST_MODEL_CHECKPOINT", "models/checkpoints/best_model.pt")
SUPPORTED_PIPELINES = {PIPELINE_VERSION, CANONICAL_PIPELINE_VERSION}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[nn.Module] = None
INPUT_LENGTH = 4
TARGET_LENGTH = 4
HIDDEN_CHANNELS: List[int] = []
MODEL_STEP_MINUTES = FORECAST_STEP_MINUTES
MODEL_GRID: Dict = {}
CURRENT_MODEL_INFO: Dict = {}
LAST_FORECAST: Dict = {}


def _lead_times_minutes(target_length: int) -> List[int]:
    return [MODEL_STEP_MINUTES * (index + 1) for index in range(target_length)]


def _confidence_by_lead(lead_times: List[int]) -> List[str]:
    values = []
    for lead in lead_times:
        if lead <= 60:
            values.append("normal_experimental")
        elif lead <= 120:
            values.append("reduced")
        elif lead <= 180:
            values.append("experimental_low_confidence")
        else:
            values.append("unsupported")
    return values


def _default_grid(pipeline_version: str) -> Dict:
    pipeline = RadarPipeline.canonical() if pipeline_version == CANONICAL_PIPELINE_VERSION else RadarPipeline()
    return pipeline.metadata()["grid"]


def _model_pipeline() -> RadarPipeline:
    if CURRENT_MODEL_INFO.get("pipeline_version") == CANONICAL_PIPELINE_VERSION:
        return RadarPipeline.canonical()
    return RadarPipeline()


def _expected_grid_shape() -> Optional[tuple[int, int]]:
    width = MODEL_GRID.get("width")
    height = MODEL_GRID.get("height")
    if width and height:
        return int(height), int(width)
    return None


def is_model_usable(model_path: str) -> bool:
    metadata = load_metadata(model_path)
    return bool(
        metadata
        and metadata.get("status") in ("completed", "published")
        and os.path.exists(os.path.join(model_path, "best_model.pt"))
    )


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_model(checkpoint_path: str) -> None:
    global model, INPUT_LENGTH, TARGET_LENGTH, HIDDEN_CHANNELS
    global MODEL_STEP_MINUTES, MODEL_GRID, CURRENT_MODEL_INFO

    if os.path.isdir(checkpoint_path):
        metadata = load_metadata(checkpoint_path)
        if metadata and not is_model_usable(checkpoint_path):
            raise ValueError(
                f"Модель {os.path.basename(checkpoint_path)} не прошла quality gate: "
                f"{metadata.get('status', 'unknown')}"
            )
        checkpoint_path = os.path.join(checkpoint_path, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found")
        return

    checkpoint = _safe_torch_load(checkpoint_path)
    pipeline_version = checkpoint.get("pipeline_version", PIPELINE_VERSION)
    if pipeline_version not in SUPPORTED_PIPELINES:
        raise ValueError(f"Неподдерживаемый pipeline модели: {pipeline_version}")

    hyperparameters = checkpoint.get("hyperparameters", {})
    INPUT_LENGTH = int(checkpoint.get("input_length", hyperparameters.get("input_length", 4)))
    TARGET_LENGTH = int(checkpoint.get("target_length", hyperparameters.get("target_length", 4)))
    HIDDEN_CHANNELS = list(checkpoint.get("hidden_channels", [32, 32]))
    MODEL_STEP_MINUTES = int(checkpoint.get("forecast_step_minutes", FORECAST_STEP_MINUTES))
    MODEL_GRID = dict(checkpoint.get("grid") or _default_grid(pipeline_version))

    loaded_model = ConvLSTM(
        input_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        kernel_size=(3, 3),
        output_steps=TARGET_LENGTH,
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()
    model = loaded_model
    CURRENT_MODEL_INFO = {
        "path": checkpoint_path,
        "model_id": os.path.basename(os.path.dirname(checkpoint_path)),
        "pipeline_version": pipeline_version,
        "model_architecture": checkpoint.get("model_architecture", "convlstm_baseline"),
        "forecast_step_minutes": MODEL_STEP_MINUTES,
        "grid": MODEL_GRID,
    }


def _resolve_registry_model_path(value: str) -> str:
    root = pathlib.Path(app.config["MODELS_REGISTRY_DIR"]).resolve()
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Разрешена загрузка моделей только из models/registry") from exc
    return str(candidate)


def _preprocess_input(array: np.ndarray) -> torch.Tensor:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 3:
        raise ValueError(f"Ожидается последовательность [T,H,W], получено {values.shape}")
    if values.shape[0] < INPUT_LENGTH:
        raise ValueError(f"Недостаточно сроков: требуется {INPUT_LENGTH}, получено {values.shape[0]}")
    values = values[-INPUT_LENGTH:]

    expected_shape = _expected_grid_shape()
    if expected_shape and values.shape[-2:] != expected_shape:
        raise ValueError(
            f"Сетка источника {values.shape[-2:]} несовместима с моделью {expected_shape}"
        )
    values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, MAX_DBZ) / MAX_DBZ
    return torch.from_numpy(values).unsqueeze(1).unsqueeze(0).float().to(device)


def _uploaded_sequence(uploaded):
    payload = np.load(uploaded.stream, allow_pickle=False)
    try:
        if isinstance(payload, np.lib.npyio.NpzFile):
            if "reflectivity" in payload:
                values = payload["reflectivity"]
            elif "arr_0" in payload:
                values = payload["arr_0"]
            else:
                raise ValueError("NPZ не содержит reflectivity или arr_0")
            if "valid_mask" in payload:
                values = np.where(payload["valid_mask"], values, 0.0)
            timestamps_raw = payload["timestamps_utc"] if "timestamps_utc" in payload else None
        else:
            values = payload
            timestamps_raw = None

        if timestamps_raw is not None:
            timestamps = [
                datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                for value in timestamps_raw
            ]
        else:
            now = datetime.datetime.now(datetime.UTC)
            timestamps = [
                now - datetime.timedelta(minutes=(len(values) - index - 1) * MODEL_STEP_MINUTES)
                for index in range(len(values))
            ]
        return np.asarray(values), timestamps
    finally:
        if isinstance(payload, np.lib.npyio.NpzFile):
            payload.close()


def _get_observed_sequence(source_type: str, station_code: str):
    grid_shape = _expected_grid_shape() or (256, 256)
    pipeline = _model_pipeline()
    if source_type == "local":
        adapter = LocalDirectoryAdapter(
            request.form.get("local_path", app.config["LOCAL_DATA_DIR"]),
            grid_size=grid_shape,
            pipeline=pipeline,
        )
        return adapter.get_latest_sequence(INPUT_LENGTH)
    if source_type == "ftp":
        if CURRENT_MODEL_INFO.get("pipeline_version") == CANONICAL_PIPELINE_VERSION:
            raise ValueError("FTP Level III ещё не унифицирован с canonical 1 км pipeline")
        return NOAAFTPAdapter(grid_size=grid_shape).get_latest_sequence(
            INPUT_LENGTH,
            station_code=station_code,
            end_file_id=request.form.get("ftp_time", "latest"),
        )
    if source_type == "aws":
        return NOAAAWSAdapter(grid_size=grid_shape, pipeline=pipeline).get_latest_sequence(
            INPUT_LENGTH,
            station_code=station_code,
        )
    if source_type == "demo":
        return DemoRadarAdapter(grid_size=grid_shape).get_latest_sequence(INPUT_LENGTH)
    raise ValueError("Неверный тип источника")


@app.route("/")
def index():
    return render_template("index.html", local_dir=app.config["LOCAL_DATA_DIR"])


@app.route("/api/sources", methods=["GET"])
def get_sources():
    return jsonify(build_default_source_registry().describe())


@app.route("/api/ftp/stations", methods=["GET"])
def get_ftp_stations():
    return jsonify(NOAAFTPAdapter().get_available_stations())


@app.route("/api/ftp/times", methods=["GET"])
def get_ftp_times():
    return jsonify(NOAAFTPAdapter().get_available_times(request.args.get("station", "kokx")))


@app.route("/api/predict", methods=["POST"])
def predict():
    global LAST_FORECAST
    if model is None:
        return jsonify({"error": "Модель ИИ не загружена"}), 500

    source_type = request.form.get("source_type", "")
    station_code = request.form.get("ftp_station", "kokx")
    try:
        if source_type == "upload":
            uploaded = request.files.get("file")
            if not uploaded:
                return jsonify({"error": "Файл не выбран"}), 400
            array, timestamps = _uploaded_sequence(uploaded)
            status_message = "Файл загружен вручную"
            source_status = "observed"
        else:
            sequence = _get_observed_sequence(source_type, station_code)
            array, timestamps, status_message = sequence
            source_status = sequence.status

        tensor_input = _preprocess_input(array)
        with torch.no_grad():
            predictions, _ = model(tensor_input)

        input_data = tensor_input.cpu().squeeze(0).squeeze(1).numpy()
        forecast_data = predictions.cpu().squeeze(0).squeeze(1).numpy()
        diagnostics = summarize_forecast(forecast_data * MAX_DBZ)
        if diagnostics["uniform_field_anomaly"]:
            return jsonify({
                "error": "Прогноз отклонён: обнаружен почти однородный слой отражаемости",
                "source_status": source_status,
                "diagnostics": diagnostics,
            }), 422

        last_time = timestamps[-1]
        range_km = float(MODEL_GRID.get("radius_km", 250.0))
        images = generate_sequence_plots(
            input_data,
            forecast_data,
            INPUT_LENGTH,
            station_code=station_code,
            start_datetime=last_time,
            history_timestamps=timestamps[-INPUT_LENGTH:],
            interval_minutes=MODEL_STEP_MINUTES,
            max_range_km=range_km,
        )

        history = [
            {
                "data": base64.b64encode(images[index]).decode("utf-8"),
                "label": timestamps[-INPUT_LENGTH + index].strftime("%H:%M") + " UTC",
            }
            for index in range(INPUT_LENGTH)
        ]
        lead_times = _lead_times_minutes(TARGET_LENGTH)
        forecast = []
        for index, lead_time in enumerate(lead_times):
            valid_time = last_time + datetime.timedelta(minutes=lead_time)
            forecast.append({
                "data": base64.b64encode(images[INPUT_LENGTH + index]).decode("utf-8"),
                "label": f"{valid_time:%H:%M} UTC (T+{lead_time} мин)",
                "lead_time_minutes": lead_time,
            })

        LAST_FORECAST = {
            "data": forecast_data * MAX_DBZ,
            "base_time": last_time,
            "station": station_code,
            "source": source_type,
            "model_id": CURRENT_MODEL_INFO.get("model_id", "unknown"),
            "model_architecture": CURRENT_MODEL_INFO.get("model_architecture", "unknown"),
            "pipeline_version": CURRENT_MODEL_INFO.get("pipeline_version", "legacy"),
            "forecast_step_minutes": MODEL_STEP_MINUTES,
        }
        return jsonify({
            "product": PRODUCT_NAME,
            "units": "dBZ",
            "base_time_utc": last_time.isoformat(),
            "forecast_step_minutes": MODEL_STEP_MINUTES,
            "horizon_minutes": lead_times[-1] if lead_times else 0,
            "lead_times_minutes": lead_times,
            "pipeline_version": CURRENT_MODEL_INFO.get("pipeline_version", "legacy"),
            "model_id": CURRENT_MODEL_INFO.get("model_id", "unknown"),
            "model_architecture": CURRENT_MODEL_INFO.get("model_architecture", "unknown"),
            "grid": MODEL_GRID,
            "history": history,
            "forecast": forecast,
            "status": status_message,
            "source_status": source_status,
            "confidence_by_lead": _confidence_by_lead(lead_times),
            "diagnostics": diagnostics,
            "warnings": ["not_official_warning", "reflectivity_only_no_nwp"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/model/load", methods=["POST"])
def load_model_route():
    model_path = request.form.get("model_path")
    if not model_path:
        return jsonify({"error": "Путь к модели не указан"}), 400
    try:
        safe_path = _resolve_registry_model_path(model_path)
        _load_model(safe_path)
        return jsonify({"success": True, "message": f"Модель {os.path.basename(safe_path)} загружена"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/inventory/raw", methods=["GET"])
def get_raw_inventory():
    return jsonify(scan_inventory(app.config["RAW_DATA_DIR"]))


@app.route("/api/inventory/datasets", methods=["GET"])
def get_datasets_inventory():
    return jsonify(scan_inventory(app.config["DATASETS_DIR"]))


@app.route("/api/inventory/models", methods=["GET"])
def get_models_inventory():
    items = scan_inventory(app.config["MODELS_REGISTRY_DIR"])
    for item in items:
        item["usable"] = is_model_usable(item["path"])
    return jsonify(items)


@app.route("/api/task/download", methods=["POST"])
@app.route("/api/task/prepare", methods=["POST"])
@app.route("/api/task/train", methods=["POST"])
def background_task_disabled():
    return jsonify({
        "success": False,
        "message": "Фоновые задачи отключены до внедрения безопасного job runner. Используйте scripts/*.sh.",
    }), 503


@app.route("/api/model/details/<model_id>", methods=["GET"])
def get_model_details(model_id):
    model_path = os.path.join(app.config["MODELS_REGISTRY_DIR"], model_id)
    metadata = load_metadata(model_path)
    if not metadata:
        return jsonify({"error": "Метаданные не найдены"}), 404
    plot_path = os.path.join(model_path, "learning_curve.png")
    plot = None
    if os.path.exists(plot_path):
        with open(plot_path, "rb") as image:
            plot = base64.b64encode(image.read()).decode("utf-8")
    return jsonify({"metadata": metadata, "plot": plot})


@app.route("/api/export/netcdf", methods=["GET"])
def export_netcdf():
    if LAST_FORECAST.get("data") is None:
        return jsonify({"error": "Прогноз не найден. Сначала запустите инференс"}), 404
    try:
        export_dir = "data/exports"
        os.makedirs(export_dir, exist_ok=True)
        station = LAST_FORECAST.get("station", "unknown")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(export_dir, f"forecast_{station}_{timestamp}.nc")
        save_forecast_to_netcdf(
            forecast_data=LAST_FORECAST["data"],
            base_time=LAST_FORECAST["base_time"],
            station_id=station,
            output_path=file_path,
            station_lon=RADAR_COORDS.get(station.lower(), (None, None))[0],
            station_lat=RADAR_COORDS.get(station.lower(), (None, None))[1],
            model_id=LAST_FORECAST.get("model_id", "unknown"),
            model_architecture=LAST_FORECAST.get("model_architecture", "unknown"),
            source=LAST_FORECAST.get("source", "unknown"),
            pipeline_version=LAST_FORECAST.get("pipeline_version", "legacy"),
            interval_minutes=LAST_FORECAST.get("forecast_step_minutes", MODEL_STEP_MINUTES),
            grid_resolution=float(MODEL_GRID.get("resolution_m", 1000.0)),
        )
        from flask import send_file
        return send_file(os.path.abspath(file_path), as_attachment=True)
    except Exception as exc:
        return jsonify({"error": f"Ошибка экспорта: {exc}"}), 500


@app.route("/api/task/logs/<task_id>", methods=["GET"])
def get_task_logs(task_id):
    return jsonify({"error": "Фоновый task runner отключён"}), 503


@app.route("/api/data/preview", methods=["GET"])
def preview_data():
    station = request.args.get("station", "KOKX").upper()
    date_value = request.args.get("date")
    if not date_value:
        return jsonify({"error": "Дата не указана"}), 400
    try:
        import nexradaws
        date = datetime.datetime.strptime(date_value, "%Y-%m-%d")
        scans = nexradaws.NexradAwsInterface().get_avail_scans(
            date.year,
            date.month,
            date.day,
            station,
        )
        counts = [0] * 24
        for scan in scans:
            if hasattr(scan, "scan_time"):
                counts[scan.scan_time.hour] += 1
        return jsonify({
            "station": station,
            "date": date_value,
            "total_scans": len(scans),
            "hourly_counts": counts,
            "has_data": bool(scans),
        })
    except ValueError:
        return jsonify({"error": "Неверный формат даты. Используйте YYYY-MM-DD"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    _load_model(CHECKPOINT_PATH)
    port = int(os.environ.get("PORT", 5005))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
