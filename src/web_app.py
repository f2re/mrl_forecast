import matplotlib
matplotlib.use("Agg")

import base64
import datetime
import os
import pathlib
import re
import sys
from typing import Dict, List

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

import numpy as np
from flask import Flask, jsonify, render_template, request
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(pathlib.Path(__file__).parent))

from adapters import DemoRadarAdapter, LocalDirectoryAdapter, NOAAAWSAdapter, NOAAFTPAdapter
from config import MAX_DBZ, PRODUCT_NAME
from diagnostic_visualization import render_evolution_layers
from dwd_source import DWDOpenDataAdapter
from export_utils import save_forecast_to_netcdf
from forecast_quality import summarize_forecast
from jobs import JobStore
from map_visualization import RADAR_COORDS, generate_sequence_plots
from metadata_utils import load_metadata, scan_inventory
from model_runtime import ModelRuntime
from radar_catalog import RadarCatalog
from radar_pipeline import CANONICAL_PIPELINE_VERSION
from source_registry import build_default_source_registry

app = Flask(__name__, template_folder="../templates")
app.config["RAW_DATA_DIR"] = str(ROOT / "data" / "raw" / "archive")
app.config["DATASETS_DIR"] = str(ROOT / "data" / "processed_archive")
app.config["MODELS_REGISTRY_DIR"] = str(ROOT / "models" / "registry")
app.config["LOCAL_DATA_DIR"] = os.environ.get("RADAR_DATA_DIR", str(ROOT / "data" / "processed"))

CHECKPOINT_PATH = os.environ.get("NOWCAST_MODEL_CHECKPOINT", str(ROOT / "models" / "checkpoints" / "best_model.pt"))
RUNTIME = ModelRuntime(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
JOB_STORE = JobStore()
CATALOG = RadarCatalog()
LAST_FORECAST: Dict = {}


def _lead_times_minutes() -> List[int]:
    return [RUNTIME.forecast_step_minutes * (index + 1) for index in range(RUNTIME.target_length)]


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


def is_model_usable(model_path: str) -> bool:
    metadata = load_metadata(model_path)
    return bool(
        metadata
        and metadata.get("status") in ("completed", "published")
        and os.path.exists(os.path.join(model_path, "best_model.pt"))
    )


def _safe_child_path(value: str, root_value: str, must_exist: bool = True) -> pathlib.Path:
    root = pathlib.Path(root_value).resolve()
    candidate = pathlib.Path(value)
    candidate = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Путь должен находиться внутри {root}") from exc
    if must_exist and not candidate.exists():
        raise ValueError(f"Путь не существует: {candidate}")
    return candidate


def _resolve_registry_model_path(value: str) -> pathlib.Path:
    return _safe_child_path(value, app.config["MODELS_REGISTRY_DIR"])


def _load_model(value: str) -> Dict:
    path = pathlib.Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    model_dir = path if path.is_dir() else path.parent
    metadata = load_metadata(str(model_dir))
    if metadata and metadata.get("status") not in ("completed", "published"):
        raise ValueError(
            f"Модель {model_dir.name} не прошла quality gate: {metadata.get('status', 'unknown')}"
        )
    checkpoint = model_dir / "best_model.pt" if path.is_dir() else path
    return RUNTIME.load(str(checkpoint))


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


def _parse_timestamp(value) -> datetime.datetime:
    text = str(value.decode("utf-8") if isinstance(value, bytes) else value)
    timestamp = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=datetime.UTC)
    return timestamp.astimezone(datetime.UTC)


def _uploaded_sequence(uploaded):
    payload = np.load(uploaded.stream, allow_pickle=False)
    inferred_time = False
    try:
        if isinstance(payload, np.lib.npyio.NpzFile):
            if "reflectivity" in payload:
                values = np.asarray(payload["reflectivity"], dtype=np.float32)
            elif "arr_0" in payload:
                values = np.asarray(payload["arr_0"], dtype=np.float32)
            else:
                raise ValueError("NPZ не содержит reflectivity или arr_0")
            masks = np.asarray(payload["valid_mask"], dtype=bool) if "valid_mask" in payload else np.isfinite(values)
            timestamps_raw = payload["timestamps_utc"] if "timestamps_utc" in payload else None
        else:
            values = np.asarray(payload, dtype=np.float32)
            masks = np.isfinite(values)
            timestamps_raw = None

        if values.ndim != 3 or masks.shape != values.shape:
            raise ValueError("Загруженный файл должен содержать reflectivity и mask формы [T,H,W]")
        if timestamps_raw is not None:
            timestamps = [_parse_timestamp(value) for value in timestamps_raw]
            if len(timestamps) != values.shape[0]:
                raise ValueError("timestamps_utc не соответствует длине sequence")
        else:
            inferred_time = True
            now = datetime.datetime.now(datetime.UTC)
            timestamps = [
                now - datetime.timedelta(
                    minutes=(len(values) - index - 1) * RUNTIME.forecast_step_minutes
                )
                for index in range(len(values))
            ]
        return values, masks, timestamps, inferred_time
    finally:
        if isinstance(payload, np.lib.npyio.NpzFile):
            payload.close()


def _get_observed_sequence(source_type: str, station_code: str):
    grid_shape = RUNTIME.expected_grid_shape() or (256, 256)
    pipeline = RUNTIME.pipeline()
    if source_type == "local":
        return LocalDirectoryAdapter(
            request.form.get("local_path", app.config["LOCAL_DATA_DIR"]),
            grid_size=grid_shape,
            pipeline=pipeline,
        ).get_latest_sequence(RUNTIME.input_length)
    if source_type == "ftp":
        if RUNTIME.info.get("pipeline_version") == CANONICAL_PIPELINE_VERSION:
            raise ValueError("FTP Level III ещё не унифицирован с canonical 1 км pipeline")
        return NOAAFTPAdapter(grid_size=grid_shape).get_latest_sequence(
            RUNTIME.input_length,
            station_code=station_code,
            end_file_id=request.form.get("ftp_time", "latest"),
        )
    if source_type == "aws":
        return NOAAAWSAdapter(grid_size=grid_shape, pipeline=pipeline).get_latest_sequence(
            RUNTIME.input_length,
            station_code=station_code,
        )
    if source_type == "demo":
        return DemoRadarAdapter(grid_size=grid_shape).get_latest_sequence(RUNTIME.input_length)
    raise ValueError("Неверный тип источника")


def _encode_images(images: List[bytes], labels: List[str]) -> List[Dict]:
    return [
        {"data": base64.b64encode(image).decode("utf-8"), "label": labels[index]}
        for index, image in enumerate(images)
    ]


def _evolution_summary(diagnostics: Dict[str, np.ndarray]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if "motion" in diagnostics:
        motion = diagnostics["motion"]
        summary["mean_motion_pixels"] = float(np.mean(np.sqrt(motion[:, 0] ** 2 + motion[:, 1] ** 2)))
    for name in ("growth", "decay", "uncertainty"):
        if name in diagnostics:
            summary[f"mean_{name}"] = float(np.mean(diagnostics[name]))
    return summary


def _enqueue_download_job() -> dict:
    source_id = request.form.get("source", "noaa-aws")
    date_value = request.form.get("date", "")
    datetime.datetime.strptime(date_value, "%Y-%m-%d")
    count = _form_int("count", 100, 1, 2000)

    if source_id == "dwd-open-data":
        station = request.form.get("station", "ess").lower()
        if not re.fullmatch(r"[a-z0-9]{3}", station):
            raise ValueError("station: для DWD ожидается трёхсимвольный код")
        command = [
            sys.executable,
            str(ROOT / "src" / "download_dwd_archive.py"),
            "--station", station,
            "--date", date_value,
            "--count", str(count),
            "--output", app.config["RAW_DATA_DIR"],
        ]
        return JOB_STORE.enqueue("download-dwd", command)

    if source_id != "noaa-aws":
        raise ValueError("Источник загрузки должен быть noaa-aws или dwd-open-data")
    station = request.form.get("station", "KOKX").upper()
    if not re.fullmatch(r"[A-Z0-9]{4}", station):
        raise ValueError("station: для NOAA ожидается четырёхсимвольный код")
    return JOB_STORE.enqueue(
        "download-noaa",
        [
            sys.executable,
            str(ROOT / "src" / "download_archive.py"),
            "--station", station,
            "--date", date_value,
            "--count", str(count),
            "--output", app.config["RAW_DATA_DIR"],
        ],
    )


def _enqueue_prepare_job() -> dict:
    archive = _safe_child_path(request.form.get("archive_dir", ""), app.config["RAW_DATA_DIR"])
    sequence_length = _form_int("seq_len", 8, 2, 48)
    time_step_minutes = _form_int("time_step_minutes", 15, 1, 60)
    grid_profile = request.form.get("grid_profile", "canonical")
    if grid_profile not in ("canonical", "legacy"):
        raise ValueError("grid_profile должен быть canonical или legacy")
    return JOB_STORE.enqueue(
        "prepare",
        [
            sys.executable,
            str(ROOT / "src" / "make_dataset.py"),
            "--archive-dir", str(archive),
            "--output-dir", app.config["DATASETS_DIR"],
            "--seq-len", str(sequence_length),
            "--grid-profile", grid_profile,
            "--time-step-minutes", str(time_step_minutes),
        ],
    )


def _enqueue_train_job() -> dict:
    raw_values = request.form.getlist("dataset_dirs[]") or request.form.getlist("dataset_dirs")
    if len(raw_values) == 1 and "," in raw_values[0]:
        raw_values = [value.strip() for value in raw_values[0].split(",") if value.strip()]
    if not raw_values:
        raise ValueError("Не выбран ни один датасет")
    datasets = [_safe_child_path(value, app.config["DATASETS_DIR"]) for value in raw_values]

    architecture = request.form.get("architecture", "phys-evolution")
    if architecture not in ("phys-evolution", "convlstm"):
        raise ValueError("Неизвестная архитектура")
    epochs = _form_int("epochs", 20, 1, 1000)
    batch_size = _form_int("batch_size", 1, 1, 64)
    input_length = _form_int("input_length", 4, 2, 24)
    target_length = _form_int("lead_time", 4, 1, 24)
    learning_rate = _form_float("lr", 1e-4, 1e-7, 1.0)
    validation_split = _form_float("val_split", 0.2, 0.05, 0.5)
    base_channels = _form_int("base_channels", 16, 4, 64)
    hidden_channels = _form_int("hidden_channels", 24, 4, 128)

    command = [
        sys.executable,
        str(ROOT / "src" / "train_nowcasting_model.py"),
        "--data-dirs", ",".join(str(path) for path in datasets),
        "--architecture", architecture,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--input-length", str(input_length),
        "--target-length", str(target_length),
        "--base-channels", str(base_channels),
        "--hidden-channels", str(hidden_channels),
        "--lr", str(learning_rate),
        "--val-split", str(validation_split),
        "--output-dir", app.config["MODELS_REGISTRY_DIR"],
    ]
    balanced = request.form.get("balanced_sampling") in ("true", "1", "on")
    if not balanced:
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


@app.route("/api/dwd/stations", methods=["GET"])
def get_dwd_stations():
    try:
        return jsonify(DWDOpenDataAdapter().list_stations())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/catalog/summary", methods=["GET"])
def get_catalog_summary():
    return jsonify(CATALOG.summary())


@app.route("/api/catalog/observations", methods=["GET"])
def get_catalog_observations():
    try:
        return jsonify(
            CATALOG.list_observations(
                source=request.args.get("source"),
                station=request.args.get("station"),
                limit=request.args.get("limit", 100),
            )
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    global LAST_FORECAST
    if not RUNTIME.loaded:
        return jsonify({"error": "Модель ИИ не загружена"}), 500

    source_type = request.form.get("source_type", "")
    station_code = request.form.get("ftp_station", "kokx")
    try:
        warnings = ["not_official_warning", "reflectivity_only_no_nwp"]
        if source_type == "upload":
            uploaded = request.files.get("file")
            if not uploaded:
                return jsonify({"error": "Файл не выбран"}), 400
            values, masks, timestamps, inferred_time = _uploaded_sequence(uploaded)
            status_message = "Файл загружен вручную"
            source_status = "observed"
            if inferred_time:
                warnings.append("manual_timestamp_inferred")
        else:
            sequence = _get_observed_sequence(source_type, station_code)
            values = sequence.stack(require_observed=sequence.status == "observed")
            masks = np.stack([frame.valid_mask for frame in sequence.frames], axis=0)
            timestamps = sequence.timestamps
            status_message = sequence.message
            source_status = sequence.status

        result = RUNTIME.predict(values, masks)
        input_data = result["input"]
        forecast_data = result["forecast"]
        diagnostics = result["diagnostics"]
        reflectivity_diagnostics = summarize_forecast(forecast_data * MAX_DBZ)
        if reflectivity_diagnostics["uniform_field_anomaly"]:
            return jsonify({
                "error": "Прогноз отклонён: обнаружен почти однородный слой отражаемости",
                "source_status": source_status,
                "diagnostics": reflectivity_diagnostics,
            }), 422

        timestamps = timestamps[-RUNTIME.input_length:]
        last_time = timestamps[-1]
        range_km = float(RUNTIME.grid.get("radius_km", 250.0))
        sequence_images = generate_sequence_plots(
            input_data,
            forecast_data,
            RUNTIME.input_length,
            station_code=station_code,
            start_datetime=last_time,
            history_timestamps=timestamps,
            interval_minutes=RUNTIME.forecast_step_minutes,
            max_range_km=range_km,
        )
        history_labels = [timestamp.strftime("%H:%M") + " UTC" for timestamp in timestamps]
        lead_times = _lead_times_minutes()
        forecast_labels = [
            f"{last_time + datetime.timedelta(minutes=lead):%H:%M} UTC (T+{lead} мин)"
            for lead in lead_times
        ]
        history = _encode_images(sequence_images[:RUNTIME.input_length], history_labels)
        forecast = _encode_images(sequence_images[RUNTIME.input_length:], forecast_labels)

        layers: Dict[str, List[Dict]] = {"reflectivity": forecast}
        rendered_diagnostics = render_evolution_layers(diagnostics, lead_times, range_km)
        for name, images in rendered_diagnostics.items():
            labels = [f"{name} · T+{lead_times[index]} мин" for index in range(len(images))]
            layers[name] = _encode_images(images, labels)

        LAST_FORECAST = {
            "data": forecast_data * MAX_DBZ,
            "base_time": last_time,
            "station": station_code,
            "source": source_type,
            "model_id": RUNTIME.info.get("model_id", "unknown"),
            "model_architecture": RUNTIME.architecture,
            "pipeline_version": RUNTIME.info.get("pipeline_version", "legacy"),
            "forecast_step_minutes": RUNTIME.forecast_step_minutes,
        }
        return jsonify({
            "product": PRODUCT_NAME,
            "units": "dBZ",
            "base_time_utc": last_time.isoformat(),
            "forecast_step_minutes": RUNTIME.forecast_step_minutes,
            "horizon_minutes": lead_times[-1] if lead_times else 0,
            "lead_times_minutes": lead_times,
            "pipeline_version": RUNTIME.info.get("pipeline_version", "legacy"),
            "model_id": RUNTIME.info.get("model_id", "unknown"),
            "model_architecture": RUNTIME.architecture,
            "grid": RUNTIME.grid,
            "history": history,
            "forecast": forecast,
            "layers": layers,
            "status": status_message,
            "source_status": source_status,
            "confidence_by_lead": _confidence_by_lead(lead_times),
            "diagnostics": reflectivity_diagnostics,
            "evolution_diagnostics": _evolution_summary(diagnostics),
            "warnings": warnings,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/model/load", methods=["POST"])
def load_model_route():
    value = request.form.get("model_path")
    if not value:
        return jsonify({"error": "Путь к модели не указан"}), 400
    try:
        path = _resolve_registry_model_path(value)
        info = _load_model(str(path))
        return jsonify({"success": True, "message": f"Модель {info['model_id']} загружена", "model": info})
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


@app.route("/api/task/catalog/rebuild", methods=["POST"])
def enqueue_catalog_rebuild():
    job = JOB_STORE.enqueue(
        "catalog-rebuild",
        [
            sys.executable,
            str(ROOT / "scripts" / "catalog.py"),
            "rebuild",
            "--raw-root", app.config["RAW_DATA_DIR"],
            "--datasets-root", app.config["DATASETS_DIR"],
        ],
    )
    return jsonify({"success": True, "task_id": job["id"], "job": job}), 202


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
            interval_minutes=LAST_FORECAST.get("forecast_step_minutes", RUNTIME.forecast_step_minutes),
            grid_resolution=float(RUNTIME.grid.get("resolution_m", 1000.0)),
        )
        from flask import send_file
        return send_file(os.path.abspath(file_path), as_attachment=True)
    except Exception as exc:
        return jsonify({"error": f"Ошибка экспорта: {exc}"}), 500


@app.route("/api/data/preview", methods=["GET"])
def preview_data():
    source_id = request.args.get("source", "noaa-aws")
    station = request.args.get("station", "KOKX")
    date_value = request.args.get("date")
    if not date_value:
        return jsonify({"error": "Дата не указана"}), 400
    try:
        date = datetime.datetime.strptime(date_value, "%Y-%m-%d")
        if source_id == "dwd-open-data":
            scans = [
                scan
                for scan in DWDOpenDataAdapter().list_scans(station.lower())
                if scan.scan_time.date() == date.date()
            ]
            station = station.upper()
        else:
            import nexradaws
            station = station.upper()
            scans = nexradaws.NexradAwsInterface().get_avail_scans(
                date.year, date.month, date.day, station
            )
        counts = [0] * 24
        for scan in scans:
            scan_time = getattr(scan, "scan_time", None)
            if scan_time is not None:
                counts[scan_time.hour] += 1
        return jsonify({
            "source": source_id,
            "station": station,
            "date": date_value,
            "total_scans": len(scans),
            "hourly_counts": counts,
            "has_data": bool(scans),
        })
    except ValueError:
        return jsonify({"error": "Неверный формат даты или код станции"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    try:
        _load_model(CHECKPOINT_PATH)
    except Exception as exc:
        print(f"Warning: model was not loaded: {exc}")
    port = int(os.environ.get("PORT", 5005))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
