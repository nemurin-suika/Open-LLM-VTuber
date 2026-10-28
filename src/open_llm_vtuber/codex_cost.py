import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

from fastapi import APIRouter
from starlette.responses import JSONResponse


KST = timezone(timedelta(hours=9))
CACHE_SECONDS = 30
QUERY_TIMEOUT = 15


def monthly_report(home: str) -> dict:
    if version("codex-usage") != "0.1.3":
        raise ValueError("Unsupported usage adapter version")
    from codex_usage.cli import build_report, parse_args

    report = build_report(
        parse_args(["--codex-home", home]), datetime.now(timezone.utc)
    )
    totals = report["totals"]
    return {
        "status": "ready",
        "period_start": report["period_start"],
        "period_end": report["period_end_inclusive"],
        "timezone": "Asia/Seoul",
        "currency": report["currency"],
        "amount": totals["estimated_cost"]
        if totals["cost_status"] == "complete"
        else totals["priced_subtotal"],
        "cost_status": totals["cost_status"],
        "has_usage": bool(report["rows"]),
        "unpriced_tokens": totals["unpriced_tokens"],
        "incomplete_requests": totals["incomplete_requests"],
        "warning_count": sum(
            count
            for name, count in report["diagnostics"].items()
            if name != "status_only_records"
        ),
        "updated_at": report["as_of"],
    }


class CodexCostService:
    def __init__(self):
        real_home = Path(os.environ.get("MIZUKI_REAL_HOME") or Path.home())
        self.home = os.path.expanduser(
            os.environ.get("VTUBER_CODEX_USAGE_HOME")
            or os.environ.get("CODEX_HOME")
            or str(real_home / ".codex")
        )
        self.cwd = os.path.expanduser(
            os.environ.get("VTUBER_CODEX_EXEC_ROOT")
            or os.environ.get("MIZUKI_AI_HOME")
            or str(real_home / "ai")
        )
        self.lock = asyncio.Lock()
        self.cached = None
        self.last_good = None
        self.checked_at = 0.0
        self.month = None

    async def _query(self) -> dict:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            self.home,
            cwd=self.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=QUERY_TIMEOUT
            )
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
        if process.returncode:
            raise RuntimeError("Usage query failed")
        result = json.loads(stdout)
        if not isinstance(result, dict) or result.get("status") != "ready":
            raise ValueError("Invalid usage response")
        return result

    async def get(self) -> dict:
        async with self.lock:
            month = datetime.now(KST).strftime("%Y-%m")
            if self.month != month:
                self.cached = self.last_good = None
                self.month = month
            if self.cached is not None and time.monotonic() - self.checked_at < CACHE_SECONDS:
                return self.cached
            try:
                self.cached = await self._query()
                self.last_good = self.cached
            except (OSError, RuntimeError, ValueError, asyncio.TimeoutError):
                self.cached = (
                    {**self.last_good, "status": "stale"}
                    if self.last_good is not None
                    else {"status": "unavailable"}
                )
            self.checked_at = time.monotonic()
            return self.cached


def init_codex_cost_route(config) -> APIRouter:
    router = APIRouter()
    service = CodexCostService()

    @router.get("/codex-usage")
    async def codex_cost():
        agent = config.character_config.agent_config
        settings = agent.agent_settings.letta_agent
        enabled = (
            agent.conversation_agent_choice == "letta_agent"
            and settings is not None
            and (settings.model or "").startswith("codex_cli/")
        )
        data = await service.get() if enabled else {"status": "disabled"}
        return JSONResponse(
            {"enabled": enabled, **data}, headers={"Cache-Control": "no-store"}
        )

    return router


if __name__ == "__main__":
    try:
        print(json.dumps(monthly_report(sys.argv[1])))
    except Exception:
        sys.exit(1)
