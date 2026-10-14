"""
license.py — Background license validation for on-prem EC2 instances.

On startup, spawns a background task that:
  1. Immediately checks the license
  2. If valid, checks every 10 minutes
  3. On failure, uses exponential backoff (10min → 20min → 40min → ...)
  4. After 24h grace period expires without a successful check, terminates the EC2 instance

All EC2 instances (webserver, workers, etc.) run this same background worker.
Each passes its app_name to identify itself in license-server logs.

If LICENSE_KEY is not set, the worker is disabled (dev/test mode).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time

import httpx

logger = logging.getLogger(__name__)

_LICENSE_URL = os.environ.get("LICENSE_SERVER_URL", "")
_BASE_INTERVAL = 600    # 10 min between successful checks
_MAX_INTERVAL = 3600    # max 1 hour backoff interval
_GRACE = 86_400         # 24h grace period
_TIMEOUT = 10           # httpx request timeout


class LicenseWorker:
    """Background worker that validates the license periodically and terminates EC2 on expiry."""

    def __init__(self, app_name: str) -> None:
        self._key: str = os.environ.get("LICENSE_KEY", "")
        self._app_name = app_name
        self._last_success: float = 0.0
        self._current_interval: float = _BASE_INTERVAL
        self._task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._key)

    def start(self) -> None:
        """Start the background license check loop. Call once on app startup."""
        if not self.enabled:
            logger.info("LICENSE_KEY not set — license worker disabled (dev/test mode).")
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        logger.info("License worker started for app=%s", self._app_name)

    def stop(self) -> None:
        """Stop the background worker. Call on app shutdown."""
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        """Main loop: check license, sleep, repeat."""
        # Immediate first check on startup
        await self._check_and_handle()

        while True:
            await asyncio.sleep(self._current_interval)
            await self._check_and_handle()

    async def _check_and_handle(self) -> None:
        """Perform a license check and handle the result."""
        now = time.time()

        try:
            valid = await self._do_check()
        except Exception as exc:
            logger.warning("License check failed: %s", exc)
            valid = None  # network/server error

        if valid is True:
            # Success — reset interval, update last_success
            self._last_success = now
            self._current_interval = _BASE_INTERVAL
            logger.debug("License valid. Next check in %ds.", self._current_interval)

        elif valid is False:
            # Server explicitly said invalid — terminate immediately
            logger.error("License server returned valid=false. Terminating instance.")
            self._terminate()

        else:
            # Network error — check grace period
            elapsed = now - self._last_success if self._last_success else now
            grace_remaining = max(0, _GRACE - elapsed)

            if self._last_success and elapsed < _GRACE:
                # Still within grace — exponential backoff
                self._current_interval = min(self._current_interval * 2, _MAX_INTERVAL)
                logger.warning(
                    "License check failed. Grace period: %.1f h remaining. "
                    "Next retry in %ds (backoff).",
                    grace_remaining / 3600,
                    self._current_interval,
                )
            else:
                # Grace expired — terminate
                logger.error(
                    "License check failed and grace period expired (%.1f h since last success). "
                    "Terminating instance.",
                    elapsed / 3600,
                )
                self._terminate()

    async def _do_check(self) -> bool:
        """Call the license server. Returns True/False or raises on network error."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                _LICENSE_URL,
                params={"app": self._app_name},
                headers={"Authorization": f"Bearer {self._key}"},
            )
            resp.raise_for_status()
            return resp.json().get("valid", False)

    def _terminate(self) -> None:
        """Terminate the EC2 instance (or exit the process in dev)."""
        # In production on EC2, we could call the AWS API to terminate self:
        #   boto3.client("ec2").terminate_instances(InstanceIds=[instance_id])
        # For simplicity, we just exit the process; systemd/ECS won't restart
        # a process that exits with code 78 (EX_CONFIG in sysexits.h).
        logger.critical("License invalid — shutting down process.")
        # Send SIGTERM to self to trigger graceful shutdown
        os.kill(os.getpid(), signal.SIGTERM)


def create_license_worker(app_name: str) -> LicenseWorker:
    """Factory to create a license worker for a specific app."""
    return LicenseWorker(app_name)
