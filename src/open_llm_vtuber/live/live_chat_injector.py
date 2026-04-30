"""
Module-level shared state that bridges the live platform (chzzk_live) with
the conversation pipeline (single_conversation).

Two channels:
- pending_chats: accumulated viewer chat messages, consumed and appended to
  user input at the start of each process_single_conversation call.
- immediate_queue: asyncio.Queue of high-priority texts (donations,
  subscriptions, timer-triggered chat batches) that trigger standalone
  process_single_conversation calls via the WebSocketHandler.

Moderation:
- user_registry: nickname → user_id_hash mapping, populated from chat events.
- restrict_callback: async callable set by ChzzkLivePlatform; called when
  the LLM includes [채팅제한 username] in its response.
"""

import asyncio
from typing import Callable, Awaitable, Dict, List, Optional, Tuple

# Buffered chat messages: (username, message)
_pending_chats: List[Tuple[str, str]] = []

# Lazy-initialized to avoid "no running event loop" errors at import time
_immediate_queue: Optional[asyncio.Queue] = None

_header: str = "[시청자 채팅]"

# nickname → user_id_hash (filled on every incoming chat)
_user_registry: Dict[str, str] = {}

# Async callable(nickname: str) -> bool, set by ChzzkLivePlatform.run()
_restrict_callback: Optional[Callable[[str], Awaitable[bool]]] = None


# ── Configuration ──────────────────────────────────────────────────────────

def configure(header: str = "[시청자 채팅]") -> None:
    global _header
    _header = header


# ── Chat buffer ─────────────────────────────────────────────────────────────

def add_chat(username: str, message: str) -> None:
    """Buffer a single viewer chat. Called from the chzzk on_chat event."""
    _pending_chats.append((username, message))


def consume_pending_chats() -> Optional[str]:
    """
    Return all buffered chats as a formatted string and clear the buffer.
    Returns None if the buffer is empty.
    Called from process_single_conversation before the LLM call.
    """
    if not _pending_chats:
        return None
    lines = "\n".join(f"{u}: {m}" for u, m in _pending_chats)
    _pending_chats.clear()
    return f"{_header}\n{lines}"


# ── Immediate-event queue ───────────────────────────────────────────────────

def get_immediate_queue() -> asyncio.Queue:
    """Return (lazily creating) the immediate-event queue."""
    global _immediate_queue
    if _immediate_queue is None:
        _immediate_queue = asyncio.Queue()
    return _immediate_queue


async def put_immediate(text: str) -> None:
    """Enqueue a high-priority event text for standalone conversation processing."""
    await get_immediate_queue().put(text)


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
