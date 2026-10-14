"""Validate outbound Jira origins to prevent SSRF via user-supplied site URLs.

Default allow is Atlassian Cloud only. Other hosts must be listed in
JIRA_ORIGIN_ALLOWLIST. Allowlisted Data Center hosts may keep a path prefix.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

_ATLASSIAN_SUFFIXES = (".atlassian.net",)
_ATLASSIAN_HOSTS = frozenset({"atlassian.net"})
_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata.google.com",
        "metadata",
    }
)
_ALLOWLIST_ENV = "JIRA_ORIGIN_ALLOWLIST"


class JiraOriginError(ValueError):
    """Raised when a Jira site URL is not a safe outbound origin."""


def _allowlist_hosts() -> list[str]:
    raw = os.environ.get(_ALLOWLIST_ENV, "").strip()
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _host_allowed(host: str, allowlist: list[str]) -> bool:
    for pattern in allowlist:
        if pattern.startswith("*."):
            suffix = pattern[1:]
            if host.endswith(suffix) or host == pattern[2:]:
                return True
            continue
        if host == pattern:
            return True
    return False


def _is_atlassian_host(host: str) -> bool:
    if host in _ATLASSIAN_HOSTS:
        return True
    return any(host.endswith(suffix) for suffix in _ATLASSIAN_SUFFIXES)


def _ip_is_public(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def normalize_jira_site_url(site_url: str) -> str:
    """Return a canonical origin (scheme://host[:port]) after SSRF checks."""
    raw = (site_url or "").strip()
    if not raw:
        raise JiraOriginError("Jira site_url is required")

    parsed = urlparse(raw)
    if parsed.scheme not in {"https", "http"}:
        raise JiraOriginError("Jira site_url must be an http(s) URL")

    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise JiraOriginError("Jira site_url must include a hostname")
    if host in _BLOCKED_HOSTS:
        raise JiraOriginError("Jira site_url host is not allowed")

    allowlist = _allowlist_hosts()
    allowlisted = _host_allowed(host, allowlist)

    if parsed.scheme != "https" and not allowlisted:
        raise JiraOriginError("Jira site_url must use HTTPS")

    if host.startswith("[") or ":" in host and not allowlisted:
        # IPv6 literals are only accepted when explicitly allowlisted.
        if not allowlisted:
            raise JiraOriginError("Jira site_url host is not allowed")

    try:
        ipaddress.ip_address(host)
        ip_literal = True
    except ValueError:
        ip_literal = False

    if ip_literal:
        if allowlisted:
            pass
        elif not _ip_is_public(host):
            raise JiraOriginError("Jira site_url must not target a private or loopback address")
        else:
            raise JiraOriginError(
                "Jira site_url must be an Atlassian Cloud host or an explicit allowlist entry"
            )
    elif allowlisted:
        pass
    elif _is_atlassian_host(host):
        pass
    else:
        raise JiraOriginError(
            "Jira site_url must be an Atlassian Cloud host or an explicit allowlist entry"
        )

    if parsed.username or parsed.password:
        raise JiraOriginError("Jira site_url must not include credentials")
    if parsed.query or parsed.fragment:
        raise JiraOriginError("Jira site_url must not include query or fragment")

    path = (parsed.path or "").strip()
    if path in {"", "/"}:
        path = ""
    else:
        parts = [part for part in path.split("/") if part]
        if ".." in parts or "\\" in path:
            raise JiraOriginError("Jira site_url path is not allowed")
        path = "/" + "/".join(parts)

    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    if allowlisted and path:
        return f"{parsed.scheme}://{netloc}{path}"
    return f"{parsed.scheme}://{netloc}"
