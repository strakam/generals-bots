"""Runner-supplied local files and presigned HTTP artifact endpoints."""

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx


def file_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
        raise ValueError("remote file authorities are unsupported")
    return Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(uri)


def read_json(uri: str) -> dict:
    scheme = urlparse(uri).scheme
    if scheme in ("http", "https"):
        response = httpx.get(uri, timeout=30, follow_redirects=True, headers={"User-Agent": "generals-coworld/0.1"})
        response.raise_for_status()
        return response.json()
    if scheme not in ("", "file"):
        raise ValueError("unsupported config URI scheme")
    return json.loads(file_path(uri).read_bytes())


def write_json(uri: str, payload: dict, method: str = "PUT") -> None:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    scheme = urlparse(uri).scheme
    if scheme in ("http", "https"):
        if method not in ("PUT", "POST"):
            raise ValueError("artifact upload method must be PUT or POST")
        response = httpx.request(
            method,
            uri,
            content=data,
            timeout=60,
            headers={"Content-Type": "application/json", "User-Agent": "generals-coworld/0.1"},
        )
        response.raise_for_status()
        return
    if scheme not in ("", "file"):
        raise ValueError("unsupported artifact URI scheme")
    path = file_path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        # Local Docker runs write into a host-owned bind mount. These public
        # episode artifacts must remain readable by the host runner's user.
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
