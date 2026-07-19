"""Open radar discovery and visual sources.

These clients do not claim that an image source contains quantitative dBZ data.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from radar_contract import RadarSourceCapabilities


class Wis2RadarCatalog:
    """Search the operational WIS2 Global Discovery Catalogue."""

    DEFAULT_ITEMS_URL = (
        "https://wis2-gdc.weather.gc.ca/collections/"
        "wis2-discovery-metadata/items"
    )
    RUSSIA_BBOX = (19.0, 41.0, 180.0, 82.0)
    CAPABILITIES = RadarSourceCapabilities(
        source_id="wis2",
        native_format="OGC API Records",
        quantitative_reflectivity=False,
        training_allowed=False,
        visualization_allowed=False,
        notes="Discovery only; every returned dataset requires separate verification.",
    )

    def __init__(
        self,
        items_url: Optional[str] = None,
        timeout_seconds: int = 20,
        session: Optional[requests.Session] = None,
    ):
        self.items_url = items_url or os.environ.get("WIS2_GDC_ITEMS_URL", self.DEFAULT_ITEMS_URL)
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def search(
        self,
        query: str,
        *,
        bbox: Optional[tuple[float, float, float, float]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"q": query, "limit": min(max(limit, 1), 1000), "f": "json"}
        if bbox is not None:
            params["bbox"] = ",".join(str(value) for value in bbox)
        response = self.session.get(self.items_url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("features", []))

    def search_russian_radar(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Find candidate radar records intersecting the Russian territory bbox."""

        records: Dict[str, Dict[str, Any]] = {}
        for query in ("weather radar", "radar reflectivity", "radar BUFR"):
            for feature in self.search(query, bbox=self.RUSSIA_BBOX, limit=limit):
                record_id = str(feature.get("id") or feature.get("properties", {}).get("id") or "")
                if record_id:
                    records[record_id] = feature
        return list(records.values())


class MeteoinfoVisualSource:
    """Download the current Meteoinfo radar animation as a visual-only product."""

    PAGE_URL = "https://meteoinfo.ru/radanim"
    CAPABILITIES = RadarSourceCapabilities(
        source_id="meteoinfo",
        native_format="GIF/PNG composite",
        quantitative_reflectivity=False,
        training_allowed=False,
        visualization_allowed=True,
        notes="Current visual composite; palette is not treated as quantitative dBZ.",
    )

    def __init__(self, timeout_seconds: int = 20, session: Optional[requests.Session] = None):
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def discover_image_url(self) -> str:
        explicit_url = os.environ.get("METEOINFO_RADAR_IMAGE_URL")
        if explicit_url:
            return explicit_url

        response = self.session.get(self.PAGE_URL, timeout=self.timeout_seconds)
        response.raise_for_status()
        candidates = re.findall(r"<img[^>]+src=[\"']([^\"']+)[\"']", response.text, flags=re.IGNORECASE)
        ranked = [
            item
            for item in candidates
            if any(token in item.lower() for token in ("rad", "phenomena", "radar"))
            and item.lower().split("?")[0].endswith((".gif", ".png", ".jpg", ".jpeg"))
        ]
        if not ranked:
            raise RuntimeError("Meteoinfo radar image was not found on the radar page")
        return urljoin(self.PAGE_URL, ranked[0])

    def fetch_latest(self, output_dir: str) -> pathlib.Path:
        image_url = self.discover_image_url()
        response = self.session.get(image_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        suffix = pathlib.Path(image_url.split("?", 1)[0]).suffix.lower() or ".gif"
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        destination = pathlib.Path(output_dir) / f"meteoinfo_radar_{timestamp}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination


class RainViewerMetadataSource:
    """Read RainViewer frame metadata without declaring tiles quantitative."""

    METADATA_URL = "https://api.rainviewer.com/public/weather-maps.json"
    CAPABILITIES = RadarSourceCapabilities(
        source_id="rainviewer",
        native_format="XYZ image tiles",
        quantitative_reflectivity=False,
        training_allowed=False,
        visualization_allowed=True,
        notes="Short visual archive; verify upstream ownership and terms before reuse.",
    )

    def __init__(self, timeout_seconds: int = 20, session: Optional[requests.Session] = None):
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def get_metadata(self) -> Dict[str, Any]:
        response = self.session.get(self.METADATA_URL, timeout=self.timeout_seconds)
        response.raise_for_status()
        return dict(response.json())

    def latest_radar_frames(self) -> List[Dict[str, Any]]:
        radar = self.get_metadata().get("radar", {})
        return list(radar.get("past", [])) + list(radar.get("nowcast", []))
