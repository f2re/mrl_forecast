import datetime
import json
import os
import pathlib
from typing import Dict, List, Optional


def save_metadata(path: str, metadata: Dict):
    """Сохраняет метаданные в JSON файл в указанной папке."""
    os.makedirs(path, exist_ok=True)
    metadata["timestamp_updated"] = datetime.datetime.now().isoformat()
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def _read_metadata_file(meta_path: pathlib.Path) -> Optional[Dict]:
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _attach_source_access(metadata: Dict) -> Dict:
    """Mark datasets from unverified adapters as unavailable for model training."""
    if metadata.get("type") != "dataset" or not metadata.get("source_path"):
        return metadata
    source_path = pathlib.Path(str(metadata["source_path"]))
    source_metadata = _read_metadata_file(source_path / "metadata.json")
    access = source_metadata.get("access") if source_metadata else None
    if not isinstance(access, dict):
        return metadata

    metadata["source_access"] = access
    allowed = access.get("training_allowed")
    if allowed is not None:
        metadata["source_training_allowed"] = bool(allowed)
    if allowed is False and metadata.get("status") == "completed":
        metadata["status"] = "unverified_source"
        metadata["training_block_reason"] = (
            "Source adapter has not passed field, geometry, QC and licence validation"
        )
    return metadata


def load_metadata(path: str) -> Optional[Dict]:
    """Загружает metadata.json и добавляет вычисляемый статус допуска источника."""
    metadata = _read_metadata_file(pathlib.Path(path) / "metadata.json")
    return _attach_source_access(metadata) if metadata else None


def scan_inventory(base_dir: str) -> List[Dict]:
    """Сканирует базовую директорию на наличие подпапок с metadata.json."""
    inventory = []
    base_path = pathlib.Path(base_dir)
    if not base_path.exists():
        return []

    for item in base_path.iterdir():
        if item.is_dir():
            metadata = load_metadata(str(item))
            if metadata:
                metadata["path"] = str(item)
                metadata["folder_name"] = item.name
                inventory.append(metadata)

    inventory.sort(key=lambda item: item.get("timestamp_updated", ""), reverse=True)
    return inventory
