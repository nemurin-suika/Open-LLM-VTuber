"""
Module-level shared state that bridges the live platform (chzzk_live) with
the conversation pipeline.

- immediate_queue: every live event (chat, donation, subscription) is enqueued
  here and triggers a standalone process_single_conversation call via the
  WebSocketHandler's background loop.
- user_registry: nickname → user_id_hash, populated on every incoming chat,
  used for moderation (temporary chat restriction).
- restrict_callback: async callable set by ChzzkLivePlatform; called when
  the LLM includes [채팅제한 username] in its response.
"""

import asyncio
import re
from typing import Callable, Awaitable, Dict, Optional

_YT_PATTERN = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/watch\?[^\s]*v=|youtu\.be/)[\w\-]+[^\s]*)"
)


def _tag_youtube_links(text: str) -> str:
    return _YT_PATTERN.sub(r"[유튜브 링크]\1[/유튜브 링크]", text)

# Lazy-initialized to avoid "no running event loop" errors at import time
_immediate_queue: Optional[asyncio.Queue] = None

# High-priority queue for donations and subscriptions
_priority_queue: Optional[asyncio.Queue] = None

# nickname → user_id_hash (filled on every incoming chat)
_user_registry: Dict[str, str] = {}

# Async callable(nickname: str) -> bool, set by ChzzkLivePlatform.run()
_restrict_callback: Optional[Callable[[str], Awaitable[bool]]] = None

# Real-time viewer count: -1 = unknown, 0+ = actual count from platform API
_viewer_count: int = -1


# ── Immediate-event queue ───────────────────────────────────────────────────

def get_immediate_queue() -> asyncio.Queue:
    """Return (lazily creating) the immediate-event queue."""
    global _immediate_queue
    if _immediate_queue is None:
        _immediate_queue = asyncio.Queue()
    return _immediate_queue


async def put_immediate(item) -> None:
    """Enqueue a regular event (chat, user input) for conversation processing.

    item can be:
    - str: treated as a live platform event (from_live=True)
    - dict: {"user_input": str|ndarray, "images": optional, "metadata": dict}
    """
    if isinstance(item, str):
        item = {"user_input": _tag_youtube_links(item), "images": None, "metadata": {"from_live": True}}
    await get_immediate_queue().put(item)


def get_priority_queue() -> asyncio.Queue:
    """Return (lazily creating) the high-priority event queue (donations, subscriptions)."""
    global _priority_queue
    if _priority_queue is None:
        _priority_queue = asyncio.Queue()
    return _priority_queue


async def put_priority(item) -> None:
    """Enqueue a high-priority event (donation, subscription) that plays before regular items.

    item can be:
    - str: treated as a live platform event (from_live=True)
    - dict: {"user_input": str|ndarray, "images": optional, "metadata": dict}
    """
    if isinstance(item, str):
        item = {"user_input": _tag_youtube_links(item), "images": None, "metadata": {"from_live": True}}
    await get_priority_queue().put(item)


# ── User registry ───────────────────────────────────────────────────────────

def register_user(nickname: str, user_id: str) -> None:
    """Store nickname → user_id_hash mapping when a chat message arrives."""
    if nickname and user_id:
        _user_registry[nickname] = user_id


def get_user_id(nickname: str) -> Optional[str]:
    """Look up a user_id_hash by nickname. Returns None if not seen yet."""
    return _user_registry.get(nickname)


# ── Moderation callbacks ────────────────────────────────────────────────────

def set_restrict_callback(cb: Callable[[str], Awaitable[bool]]) -> None:
    """Register the async callable that executes a temporary chat restriction."""
    global _restrict_callback
    _restrict_callback = cb


async def execute_restrict(nickname: str) -> bool:
    """Call the platform's restrict callback for the given nickname."""
    if _restrict_callback:
        return await _restrict_callback(nickname)
    return False


# ── Viewer count ─────────────────────────────────────────────────────────────

def set_viewer_count(count: int) -> None:
    """Update the real-time viewer count."""
    global _viewer_count
    _viewer_count = count


def get_viewer_count() -> int:
    """Return the last known viewer count. -1 means not yet fetched."""
    return _viewer_count
