"""Cached, rate-limited HTTP access to public scientific APIs.

Every fetch returns a :class:`Fetch` carrying the exact URL, the retrieval
timestamp, a SHA-256 of the response body and whether the body came from cache.
Nothing in ATLAS is allowed to parse a response without that envelope, so any
stored conclusion can be traced to the bytes it was derived from.

Rate limits are enforced per host against the limits each provider publishes.
The cache is on disk and content-addressed by request URL, which makes repeated
analysis runs reproducible and keeps development off the providers' servers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode, urlparse

import requests

logger = logging.getLogger(__name__)

__all__ = ["Fetch", "FetchError", "RateLimit", "CachedHttpClient", "DEFAULT_CACHE_DIR"]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Contact string is sent in User-Agent. NCBI asks for an identifying agent and
# will throttle or block anonymous high-volume traffic.
USER_AGENT = (
    "ATLAS-research-platform/0.1 (computational pharmacology research; "
    "contact: set ATLAS_CONTACT_EMAIL)"
)


class FetchError(RuntimeError):
    """A request failed after exhausting retries, or returned an error status."""

    def __init__(self, message: str, *, url: str, status: int | None = None):
        super().__init__(message)
        self.url = url
        self.status = status


@dataclass(frozen=True, slots=True)
class Fetch:
    """A retrieved HTTP payload with provenance."""

    url: str
    status: int
    body: bytes
    fetched_at: datetime
    from_cache: bool
    content_type: str | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.body)
        except json.JSONDecodeError as exc:
            raise FetchError(
                f"response was not valid JSON: {exc}", url=self.url, status=self.status
            ) from exc


@dataclass
class RateLimit:
    """Minimum seconds between requests to one host, enforced process-wide."""

    min_interval_s: float
    _last: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval_s - (now - self._last)
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._last = now


# Published provider limits. NCBI allows 3 requests/second without an API key;
# PubChem asks for no more than 5/second; EBI and ClinicalTrials.gov do not
# publish a hard figure, so a conservative interval is used.
_HOST_LIMITS: dict[str, RateLimit] = {
    "eutils.ncbi.nlm.nih.gov": RateLimit(0.34),
    "pubchem.ncbi.nlm.nih.gov": RateLimit(0.21),
    "www.ebi.ac.uk": RateLimit(0.25),
    "rest.uniprot.org": RateLimit(0.25),
    "clinicaltrials.gov": RateLimit(0.25),
}
_DEFAULT_LIMIT = RateLimit(0.5)
_LIMIT_LOCK = threading.Lock()


def _limit_for(host: str) -> RateLimit:
    with _LIMIT_LOCK:
        if host not in _HOST_LIMITS:
            _HOST_LIMITS[host] = RateLimit(_DEFAULT_LIMIT.min_interval_s)
        return _HOST_LIMITS[host]


class CachedHttpClient:
    """HTTP client with a disk cache, per-host rate limiting and retries."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        *,
        ttl: timedelta | None = timedelta(days=30),
        timeout_s: float = 30.0,
        max_retries: int = 3,
        offline: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.offline = offline
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    # -- cache --------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _read_cache(self, url: str) -> Fetch | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("discarding unreadable cache entry for %s", url)
            return None
        fetched_at = datetime.fromisoformat(record["fetched_at"])
        if self.ttl is not None:
            if datetime.now(timezone.utc) - fetched_at > self.ttl:
                return None
        return Fetch(
            url=record["url"],
            status=record["status"],
            body=bytes.fromhex(record["body_hex"]),
            fetched_at=fetched_at,
            from_cache=True,
            content_type=record.get("content_type"),
        )

    def _write_cache(self, fetch: Fetch) -> None:
        path = self._cache_path(fetch.url)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "url": fetch.url,
            "status": fetch.status,
            "content_type": fetch.content_type,
            "fetched_at": fetch.fetched_at.isoformat(),
            "sha256": fetch.sha256,
            "body_hex": fetch.body.hex(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    # -- fetching -----------------------------------------------------------
    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        refresh: bool = False,
        accept: str | None = None,
    ) -> Fetch:
        """GET ``url``, returning a cached copy when one is valid."""
        full_url = f"{url}?{urlencode(dict(params), doseq=True)}" if params else url

        if not refresh:
            cached = self._read_cache(full_url)
            if cached is not None:
                return cached

        if self.offline:
            raise FetchError(
                "client is offline and no valid cache entry exists", url=full_url
            )

        host = urlparse(full_url).netloc
        limit = _limit_for(host)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            limit.wait()
            headers = {"Accept": accept} if accept else None
            try:
                resp = self._session.get(
                    full_url, timeout=self.timeout_s, headers=headers
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "request error for %s (attempt %d/%d): %s",
                    full_url,
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(min(2.0**attempt, 8.0))
                continue

            # 429/5xx are transient; back off and retry.
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_error = FetchError(
                    f"transient status {resp.status_code}",
                    url=full_url,
                    status=resp.status_code,
                )
                retry_after = resp.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(2.0**attempt, 8.0)
                )
                logger.warning(
                    "status %d for %s, retrying in %.1fs", resp.status_code, full_url, delay
                )
                time.sleep(delay)
                continue

            fetch = Fetch(
                url=full_url,
                status=resp.status_code,
                body=resp.content,
                fetched_at=datetime.now(timezone.utc),
                from_cache=False,
                content_type=resp.headers.get("Content-Type"),
            )

            if resp.status_code == 404:
                # A genuine "not found" is a real answer; cache it so repeated
                # resolution attempts for an unknown compound stay cheap.
                self._write_cache(fetch)
                return fetch

            if not resp.ok:
                raise FetchError(
                    f"HTTP {resp.status_code} for {full_url}",
                    url=full_url,
                    status=resp.status_code,
                )

            self._write_cache(fetch)
            return fetch

        raise FetchError(
            f"exhausted {self.max_retries} attempts: {last_error}", url=full_url
        )

    def get_json(
        self, url: str, params: Mapping[str, Any] | None = None, **kw: Any
    ) -> tuple[Any, Fetch]:
        fetch = self.get(url, params, accept="application/json", **kw)
        return fetch.json(), fetch
