import os
import json
import datetime
import pathlib
from typing import Dict, List, Optional

def save_metadata(path: str, metadata: Dict):
    """Сохраняет метаданные в JSON файл в указанной папке."""
    os.makedirs(path, exist_ok=True)
    metadata['timestamp_updated'] = datetime.datetime.now().isoformat()
    with open(os.path.join(path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

def load_metadata(path: str) -> Optional[Dict]:
    """Загружает метаданные из JSON файла."""
    meta_path = os.path.join(path, 'metadata.json')
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def scan_inventory(base_dir: str) -> List[Dict]:
    """Сканирует базовую директорию на наличие подпапок с metadata.json."""
    inventory = []
    base_path = pathlib.Path(base_dir)
    if not base_path.exists():
        return []
    
    for item in base_path.iterdir():
        if item.is_dir():
            meta = load_metadata(str(item))
            if meta:
                meta['path'] = str(item)
                meta['folder_name'] = item.name
                inventory.append(meta)
    
    # Сортировка по времени обновления (свежие сверху)
    inventory.sort(key=lambda x: x.get('timestamp_updated', ''), reverse=True)
    return inventory
