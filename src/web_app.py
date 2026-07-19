import matplotlib
matplotlib.use("Agg")

import base64
import datetime
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

import numpy as np
from flask import Flask, jsonify, render_template, request
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(pathlib.Path(__file__).parent))

from adapters import DemoRadarAdapter, LocalDirectoryAdapter, NOAAAWSAdapter, NOAAFTPAdapter
from config import FORECAST_STEP_MINUTES, MAX_DBZ, PRODUCT_NAME
from convlstm import ConvLSTM
from export_utils import save_forecast_to_netcdf
from forecast_quality import summarize_forecast
from jobs import JobStore
from map_visualization import RADAR_COORDS, generate_sequence_plots
from metadata_utils import load_metadata, scan_inventory
from radar_pipeline import CANONICAL_PIPELINE_VERSION, PIPELINE_VERSION, RadarPipeline
from source_registry import build_default_source_registry

app = Flask(__name__, template_folder="../templates")
app.config["RAW_DATA_DIR"] = str(ROOT / "data" / "raw" / "archive")
app.config["DATASETS_DIR"] = str(ROOT / "data" / "processed_archive")
app.config["MODELS_REGISTRY_DIR"] = str(ROOT / "models" / "registry")
app.config["LOCAL_DATA_DIR"] = os.environ.get("RADAR_DATA_DIR", str(ROOT / "data" / "processed"))

CHECKPOINT_PATH = os.environ.get("NOWCAST_MODEL_CHECKPOINT", str(ROOT / "models" / "checkpoints" / "best_model.pt"))
SUPPORTED_PIPELINES = {PIPELINE_VERSION, CANONICAL_PIPELINE_VERSION}
JOB_STORE = JobStore()

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

    path = pathlib.Path(checkpoint_path)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        metadata = load_metadata(str(path))
        if metadata and not is_model_usable(str(path)):
            raise ValueError(
                f"Модель {path.name} не прошла quality gate: {metadata.get('status', 'unknown')}"
            )
        path = path / "best_model.pt"

    if not path.exists():
        print(f"Warning: Checkpoint {path} not found")
        return

    checkpoint = _safe_torch_load(str(path))
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
        "path": str(path),
        "model_id": path.parent.name,
        "pipeline_version": pipeline_version,
        "model_architecture": checkpoint.get("model_architecture", "convlstm_baseline"),
        "forecast_step_minutes": MODEL_STEP_MINUTES,
        "grid": MODEL_GRID,
    }


def _safe_child_path(value: str, root_value: str, must_exist: bool = True) -> pathlib.Path:
    root = pathlib.Path(root_value).resolve()
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Путь должен находиться внутри {root}") from exc
    if must_exist and not candidate.exists():
        raise ValueError(f"Путь не существует: {candidate}")
    return candidate


def _resolve_registry_model_path(value: str) -> str:
    return str(_safe_child_path(value, app.config["MODELS_REGISTRY_DIR"]))


def _form_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(request.form.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: ожидается целое число") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name}: допустим диапазон {minimum}..{maximum}")
    return value


def _form_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: ожидается число") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name}: допустим диапазон {minimum}..{maximum}")
    return value


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


def _enqueue_download_job() -> dict:
    station = request.form.get("station", "KOKX").upper()
    if not re.fullmatch(r"[A-Z0-9]{4}", station):
        raise ValueError("station: ожидается четырёхсимвольный код")
    date_value = request.form.get("date", "")
    datetime.datetime.strptime(date_value, "%Y-%m-%d")
    count = _form_int("count", 100, 1, 2000)
    command = [
        sys.executable,
        str(ROOT / "src" / "download_archive.py"),
        "--station", station,
        "--date", date_value,
        "--count", str(count),
        "--output", app.config["RAW_DATA_DIR"],
    ]
    return JOB_STORE.enqueue("download", command)


def _enqueue_prepare_job() -> dict:
    archive = _safe_child_path(
        request.form.get("archive_dir", ""),
        app.config["RAW_DATA_DIR"],
    )
    sequence_length = _form_int("seq_len", 8, 2, 48)
    grid_profile = request.form.get("grid_profile", "canonical")
    if grid_profile not in ("canonical", "legacy"):
        raise ValueError("grid_profile должен быть canonical или legacy")
    command = [
        sys.executable,
        str(ROOT / "src" / "make_dataset.py"),
        "--archive-dir", str(archive),
        "--output-dir", app.config["DATASETS_DIR"],
        "--seq-len", str(sequence_length),
        "--grid-profile", grid_profile,
    ]
    return JOB_STORE.enqueue("prepare", command)


def _enqueue_train_job() -> dict:
    raw_values = request.form.getlist("dataset_dirs[]") or request.form.getlist("dataset_dirs")
    if len(raw_values) == 1 and "," in raw_values[0]:
        raw_values = [value.strip() for value in raw_values[0].split(",") if value.strip()]
    if not raw_values:
        raise ValueError("Не выбран ни один датасет")
    datasets = [
        _safe_child_path(value, app.config["DATASETS_DIR"])
        for value in raw_values
    ]
    epochs = _form_int("epochs", 10, 1, 1000)
    batch_size = _form_int("batch_size", 1, 1, 64)
    input_length = _form_int("input_length", 4, 2, 24)
    target_length = _form_int("lead_time", 4, 1, 24)
    learning_rate = _form_float("lr", 1e-4, 1e-7, 1.0)
    validation_split = _form_float("val_split", 0.2, 0.05, 0.5)
    command = [
        sys.executable,
        str(ROOT / "src" / "train_nowcasting_model.py"),
        "--data-dirs", ",".join(str(path) for path in datasets),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--input-length", str(input_length),
        "--target-length", str(target_length),
        "--lr", str(learning_rate),
        "--val-split", str(validation_split),
        "--output-dir", app.config["MODELS_REGISTRY_DIR"],
    ]
    if request.form.get("balanced_sampling", "true").lower() in ("false", "0", "off"):
        command.append("--no-balanced-sampling")
    return JOB_STORE.enqueue("train", command)


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


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    return jsonify(JOB_STORE.list(request.args.get("limit", 50)))


@app.route("/api/task/download", methods=["POST"])
def enqueue_download():
    try:
        job = _enqueue_download_job()
        return jsonify({"success": True, "task_id": job["id"], "job": job}), 202
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/task/prepare", methods=["POST"])
def enqueue_prepare():
    try:
        job = _enqueue_prepare_job()
        return jsonify({"success": True, "task_id": job["id"], "job": job}), 202
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/task/train", methods=["POST"])
def enqueue_train():
    try:
        job = _enqueue_train_job()
        return jsonify({"success": True, "task_id": job["id"], "job": job}), 202
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/task/logs/<task_id>", methods=["GET"])
def get_task_logs(task_id):
    try:
        job = JOB_STORE.get(task_id)
        return jsonify({**job, "logs": JOB_STORE.read_log(task_id)})
    except KeyError:
        return jsonify({"error": "Задание не найдено"}), 404


@app.route("/api/jobs/<task_id>/cancel", methods=["POST"])
def cancel_job(task_id):
    try:
        return jsonify({"success": True, "job": JOB_STORE.request_cancel(task_id)})
    except KeyError:
        return jsonify({"error": "Задание не найдено"}), 404


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
        export_dir = str(ROOT / "data" / "exports")
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
