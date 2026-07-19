"""Common access helpers for downloadable radar sources.

Credentials are read from environment variables first and from a user-local JSON
file second. Secret values are never written into the repository or returned by
probe results.
"""

from __future__ import annotations

import datetime
import getpass
import hashlib
import json
import os
import pathlib
import stat
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from radar_contract import RadarSourceCapabilities


DEFAULT_CREDENTIAL_FILE = (
    pathlib.Path.home() / ".config" / "mrl_forecast" / "credentials.json"
)


class SourceAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteRadarFile:
    source_id: str
    file_id: str
    filename: str
    url: str
    native_format: str
    timestamp_utc: Optional[datetime.datetime] = None
    station_id: str = ""
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "file_id": self.file_id,
            "filename": self.filename,
            "url": self.url,
            "native_format": self.native_format,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
            "station_id": self.station_id,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SourceProbeResult:
    source_id: str
    status: str
    reachable: bool
    can_list: bool
    can_download: bool
    credential_state: str
    message: str = ""
    sample: Optional[Dict[str, Any]] = None
    checked_at_utc: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "status": self.status,
            "reachable": self.reachable,
            "can_list": self.can_list,
            "can_download": self.can_download,
            "credential_state": self.credential_state,
            "message": self.message,
            "sample": self.sample,
            "checked_at_utc": self.checked_at_utc,
        }


class CredentialStore:
    """Store API keys outside the project tree with restrictive permissions."""

    def __init__(self, path: Optional[str] = None):
        configured = path or os.environ.get("MRL_CREDENTIALS_FILE")
        self.path = pathlib.Path(configured).expanduser() if configured else DEFAULT_CREDENTIAL_FILE

    def _load(self) -> Dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceAccessError(f"Cannot read credential store {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SourceAccessError(f"Credential store {self.path} must contain a JSON object")
        return {
            str(key): str(value)
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def get(self, env_name: str) -> Optional[str]:
        if not env_name:
            return None
        environment_value = os.environ.get(env_name)
        if environment_value:
            return environment_value
        return self._load().get(env_name)

    def set(self, env_name: str, value: str) -> pathlib.Path:
        if not env_name:
            raise ValueError("Credential environment variable is not defined")
        secret = value.strip()
        if not secret:
            raise ValueError("Credential value must not be empty")
        payload = self._load()
        payload[env_name] = secret
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        temporary.replace(self.path)
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        os.environ[env_name] = secret
        return self.path

    def configure(
        self,
        capabilities: RadarSourceCapabilities,
        value: Optional[str] = None,
    ) -> pathlib.Path:
        if not capabilities.credential_env:
            raise ValueError(f"Source {capabilities.source_id} does not use a stored API key")
        secret = value
        if secret is None:
            secret = getpass.getpass(
                f"Введите {capabilities.credential_env} для {capabilities.source_id}: "
            )
        return self.set(capabilities.credential_env, secret)

    def state(self, capabilities: RadarSourceCapabilities) -> str:
        if not capabilities.credential_env:
            return "not_required"
        return "present" if self.get(capabilities.credential_env) else "missing"


def parse_utc(value: Any) -> Optional[datetime.datetime]:
    if value is None or value == "":
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        timestamp = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=datetime.UTC)
    return timestamp.astimezone(datetime.UTC)


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_http(
    remote: RemoteRadarFile,
    output_dir: str,
    *,
    session: Optional[requests.Session] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    client = session or requests.Session()
    destination_dir = pathlib.Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / pathlib.PurePath(remote.filename).name
    digest = hashlib.sha256()
    size = 0
    with client.get(
        remote.url,
        headers=headers or {},
        timeout=timeout_seconds,
        stream=True,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    return {
        "path": str(destination.resolve()),
        "filename": destination.name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "source": remote.source_id,
        "file_id": remote.file_id,
    }


def verify_http_download(
    remote: RemoteRadarFile,
    *,
    session: Optional[requests.Session] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_seconds: int = 30,
) -> bool:
    """Read a small byte range without persisting the remote file."""

    client = session or requests.Session()
    probe_headers = dict(headers or {})
    probe_headers["Range"] = "bytes=0-1023"
    with client.get(
        remote.url,
        headers=probe_headers,
        timeout=timeout_seconds,
        stream=True,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                return True
    return False


def unsigned_s3_client(region: str, endpoint_url: Optional[str] = None):
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise SourceAccessError("boto3 and botocore are required for open S3 radar sources") from exc
    s3_config = {"addressing_style": "path"} if endpoint_url else {}
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        config=Config(signature_version=UNSIGNED, s3=s3_config),
    )


def list_s3_files(
    *,
    source_id: str,
    bucket: str,
    region: str,
    endpoint_url: Optional[str] = None,
    public_base_url: Optional[str] = None,
    prefix: str = "",
    limit: int = 20,
    native_format: str = "HDF5",
) -> list[RemoteRadarFile]:
    client = unsigned_s3_client(region, endpoint_url)
    response = client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=max(1, min(int(limit), 1000)),
    )
    result = []
    for item in response.get("Contents", []):
        key = str(item["Key"])
        if key.endswith("/"):
            continue
        if public_base_url:
            url = f"{public_base_url.rstrip('/')}/{key}"
        elif endpoint_url:
            url = f"{endpoint_url.rstrip('/')}/{bucket}/{key}"
        else:
            url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        result.append(
            RemoteRadarFile(
                source_id=source_id,
                file_id=key,
                filename=pathlib.PurePosixPath(key).name,
                url=url,
                native_format=native_format,
                timestamp_utc=parse_utc(item.get("LastModified")),
                size_bytes=int(item.get("Size", 0)),
                metadata={
                    "bucket": bucket,
                    "key": key,
                    "region": region,
                    "endpoint_url": endpoint_url,
                },
            )
        )
    return result


def verify_s3_download(remote: RemoteRadarFile) -> bool:
    metadata = remote.metadata
    bucket = metadata.get("bucket")
    key = metadata.get("key")
    region = metadata.get("region")
    if not bucket or not key or not region:
        raise SourceAccessError("S3 remote file does not contain bucket/key/region metadata")
    client = unsigned_s3_client(str(region), metadata.get("endpoint_url"))
    response = client.get_object(Bucket=str(bucket), Key=str(key), Range="bytes=0-1023")
    return bool(response["Body"].read(1024))


def download_s3_file(remote: RemoteRadarFile, output_dir: str) -> Dict[str, Any]:
    metadata = remote.metadata
    bucket = metadata.get("bucket")
    key = metadata.get("key")
    region = metadata.get("region")
    if not bucket or not key or not region:
        raise SourceAccessError("S3 remote file does not contain bucket/key/region metadata")
    destination_dir = pathlib.Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / pathlib.PurePosixPath(remote.filename).name
    client = unsigned_s3_client(str(region), metadata.get("endpoint_url"))
    client.download_file(str(bucket), str(key), str(destination))
    return {
        "path": str(destination.resolve()),
        "filename": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "source": remote.source_id,
        "file_id": remote.file_id,
    }
