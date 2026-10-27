from typing import Any, Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import os
import time
import uuid
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)
from .conversations.single_conversation import process_single_conversation
from .message_handler import message_handler
from .tool_approval import ToolApprovalManager
from . import session_log


# ── 턴 완료 마커 ──────────────────────────────────────────────────────────────
# 안전 재기동(_mz_setup/restart-stack.sh)이 "미즈키가 말을 다 전했는지"를
# 판정하는 신호. claude -p 프로세스의 종료는 턴 종료가 아니다. 그 뒤에
# letta → OLV → TTS 합성 → 프론트 전송 → 재생이 남아 있어서, claude 소멸
# 직후 스택을 종료하면 발화가 통째로 유실된다(2026-10-24 01:31 실제 발생).
#
#   delivered : captured replay 완료 = 오디오 payload 가 프론트로 전부 전송됨
#   played    : 프론트가 frontend-playback-complete 를 보냄 = 재생까지 끝남
_TURN_MARK_FILE = os.environ.get("MIZUKI_TURN_MARK_FILE", "/tmp/mizuki_turn_done")


def mark_turn_stage(stage: str) -> None:
    """턴 진행 단계를 마커 파일에 기록한다. 실패해도 대화에 영향 없음."""
    try:
        tmp = f"{_TURN_MARK_FILE}.tmp"
        with open(tmp, "w") as f:
            f.write(f"{stage} {time.time():.3f}\n")
        os.replace(tmp, _TURN_MARK_FILE)
    except Exception as e:  # pragma: no cover - 마커는 보조 기능
        logger.debug(f"[TurnMark] {stage} 기록 실패: {e}")


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]


def _filter_images_for_screen_capture_setting(
    images: Optional[List[Dict[str, Any]]],
    screen_capture_enabled: bool,
    screen_sharing_enabled: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Keep uploads and explicitly allowed shared screens under capture restrictions."""
    if not images:
        return None
    if screen_capture_enabled:
        return images

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for image in images:
        source = str(image.get("source", "")).lower() if isinstance(image, dict) else ""
        if source == "upload" or (source == "screen" and screen_sharing_enabled):
            kept.append(image)
        else:
            dropped += 1

    if dropped or kept:
        logger.info(
            "[ImageInput] enable_screen_capture=false: "
            f"enable_screen_sharing={screen_sharing_enabled}, "
            f"blocked={dropped}, allowed={len(kept)}"
        )
    return kept or None


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}

        # Claude 도구 사용 승인 중계기 (hook → /tool-approval → 프론트 → 여기서 resolve)
        self.tool_approval = ToolApprovalManager()

        # ── 다중 세션 동시 처리 (TODO19 3단계) ────────────────────────────
        # 세션별로 돌고 있는 LLM+TTS capture task (인터럽트 시 cancel 대상).
        # 세션 하나가 작업 중일 때 다른 세션에서 말을 걸어도 서로 안 끊긴다.
        self._capture_tasks: Dict[str, asyncio.Task] = {}
        # 세션별 턴 시작 시각 (epoch). /session-status 에서 "얼마나 오래 돌고 있는지"를
        # 알려주기 위한 용도. 재기동 전에 남의 작업을 끊지 않으려고 쓴다(TODO26).
        self._turn_started: Dict[str, float] = {}
        # 비활성 세션에서 완성된 발화를 모아두는 버퍼. 값은 턴 단위 메시지 묶음 목록.
        # 사용자가 그 세션으로 넘어가면 꺼내서 재생한다(마지막 턴만 음성).
        self._session_buffers: Dict[str, List[List[str]]] = {}
        # 비활성 세션 버퍼를 재생 큐로 되돌리기 위한 참조 (파이프라인 기동 시 주입)
        self._regular_pq: Optional[asyncio.Queue] = None
        # 지금 화면으로 흘려보내고 있는(=라이브 재생 중인) 세션 id 집합.
        # 재생 도중 세션이 바뀌면 남은 발화를 보류 버퍼로 되돌리는데, 그 되돌림이
        # 끝나기 전에 replay-session-buffer 요청이 들어오면 빈 버퍼를 보고 지나칠 수
        # 있다. 그 경합을 막기 위해 "아직 정리 중"임을 표시해둔다.
        self._live_replays: set = set()

        # ── 작업 중 세션의 입력 큐잉 (TODO25) ──────────────────────────────
        # 같은 세션에서 턴이 돌고 있을 때 들어온 입력. 끼어들어 LLM 을 동시에
        # 때리지 않고 여기 쌓아뒀다가, 턴이 끝나는 즉시 후속 작업으로 이어 돌린다.
        # 값은 {"id", "ts", "item"} 목록.
        self._pending_inputs: Dict[str, List[dict]] = {}
        # 대기분을 실제 턴으로 올려주는 함수 (파이프라인 기동 시 주입)
        self._start_turn: Optional[Callable] = None
        # 인터럽트 직후처럼 자동 소화를 한 번 건너뛸 세션
        self._skip_drain: set = set()

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    # ── 다중 세션 헬퍼 ────────────────────────────────────────────────────

    def _active_session_id(self) -> str:
        """사용자가 지금 보고 있는 세션 id."""
        for ctx in self.client_contexts.values():
            agent = getattr(ctx, "agent_engine", None)
            active = getattr(agent, "active_conversation_id", None)
            if active:
                return str(active)
        agent = getattr(self.default_context_cache, "agent_engine", None)
        return str(getattr(agent, "active_conversation_id", "default") or "default")

    def _any_agent(self) -> Optional[Any]:
        """세션 메타를 물어볼 수 있는 agent 인스턴스 하나."""
        for ctx in self.client_contexts.values():
            agent = getattr(ctx, "agent_engine", None)
            if agent is not None:
                return agent
        return getattr(self.default_context_cache, "agent_engine", None)

    async def resolve_session_label(self, session_key: str) -> tuple:
        """도구 승인 hook 이 보낸 세션 키 → (conversation_id, 세션 이름).

        승인 팝업에 "어느 방에서 올라온 요청인지" 적어주기 위한 변환.
        모르는 키거나 해석에 실패하면 활성 세션 기준으로 돌려준다.
        """
        agent = self._any_agent()
        resolver = getattr(agent, "resolve_session_by_key", None)
        if session_key and callable(resolver):
            try:
                return await resolver(session_key)
            except Exception as e:
                logger.debug(f"[ToolApproval] 세션 키 해석 실패 (key={session_key}): {e}")
        active = self._active_session_id()
        name = ""
        if callable(resolver):
            try:
                _, name = await resolver(active)
            except Exception:
                name = ""
        return active, name

    @staticmethod
    def _mark_replay_only(messages: List[str]) -> List[str]:
        """자막을 채팅 기록에 또 쌓지 않도록 표시한 메시지 목록.

        비활성 세션 발화는 letta 에 이미 저장되어 있어서, 세션으로 넘어오면
        fetch-letta-history 복원으로 글이 먼저 채워진다. 그 뒤에 음성을 재생할 때
        같은 문장을 다시 채팅에 추가하면 중복되므로, 프론트가 기록 추가를
        건너뛰고 소리와 입모양만 처리하게 플래그를 달아준다.
        """
        out: List[str] = []
        for raw in messages:
            try:
                data = json.loads(raw)
            except Exception:
                out.append(raw)
                continue
            if data.get("type") == "audio":
                data["replay_only"] = True
                out.append(json.dumps(data))
                continue
            out.append(raw)
        return out

    def _buffer_session_turn(
        self, session_id: str, messages: List[str], front: bool = False
    ) -> None:
        """비활성 세션의 완성된 턴을 보류 버퍼에 쌓는다.

        front=True 는 "재생 중간에 끊겨서 돌려놓는 남은 조각"처럼 이미 쌓여 있는
        턴들보다 시간상 앞선 발화일 때 쓴다 (버퍼 맨 앞에 넣는다).
        """
        if not messages:
            return
        bucket = self._session_buffers.setdefault(session_id, [])
        if front:
            bucket.insert(0, list(messages))
        else:
            bucket.append(list(messages))
        logger.info(
            f"[session] 비활성 세션 발화 보류 (session={session_id}, "
            f"대기 {len(self._session_buffers[session_id])}턴)"
        )

    async def _notify_session_activity(self, session_id: str) -> None:
        """레일에 미확인 표시를 띄워달라고 프론트에 알린다.

        count = 그 세션에 쌓인 미즈키 발화 덩어리(턴) 개수. 중간 알림은 세지 않는다
        (네무린 선택, 2026-10-25). 프론트는 이 숫자를 레일 배지로 띄운다.
        """
        payload = json.dumps(
            {
                "type": "session-activity",
                "session_id": session_id,
                "count": len(self._session_buffers.get(session_id, [])),
            }
        )
        for ws in list(self.client_connections.values()):
            try:
                await ws.send_text(payload)
            except Exception:
                pass

    # ── 작업 중 입력 큐잉 (TODO25) ───────────────────────────────────────

    # 연달아 적은 입력으로 볼 간격. 이 안에 들어온 텍스트는 한 턴으로 합친다.
    _QUEUE_MERGE_SLACK = 90.0
    # 한 세션에 쌓아둘 수 있는 대기 입력 상한 (넘치면 가장 오래된 것부터 버린다)
    _QUEUE_MAX_ITEMS = 20

    def _session_busy(self, session_id: str) -> bool:
        """그 세션이 지금 턴을 돌고 있는지."""
        task = self._capture_tasks.get(session_id)
        return task is not None and not task.done()

    # ── 재기동 충돌 방지용 상태 조회 (TODO26) ────────────────────────────

    def session_status(self) -> dict:
        """지금 돌고 있는 세션이 있는지 한눈에 보여주는 스냅샷.

        미즈키가 letta/백엔드를 재기동하기 전에 이걸 먼저 읽고, 다른 세션이
        작업 중이면 멈춘다. 재기동은 돌고 있던 턴을 통째로 날리기 때문에
        반만 고쳐진 파일이 남을 수 있다.

        백엔드가 죽어 있으면 조회 자체가 실패하는데, 그건 돌고 있는 턴도
        없다는 뜻이라 그대로 재기동해도 안전하다.
        """
        now = time.time()
        agent = self._any_agent()
        names = dict(getattr(agent, "_session_names", {}) or {})
        active = self._active_session_id()

        known = set(names) | set(self._capture_tasks) | set(self._pending_inputs)
        known.add(active)

        sessions = []
        for sid in sorted(known):
            busy = self._session_busy(sid)
            started = self._turn_started.get(sid)
            sessions.append(
                {
                    "id": sid,
                    "name": (names.get(sid) or "").strip(),
                    "busy": busy,
                    "active": sid == active,
                    "elapsed_sec": (
                        round(now - started, 1) if busy and started else None
                    ),
                    "queued": len(self._pending_inputs.get(sid) or []),
                    "unread_turns": len(self._session_buffers.get(sid) or []),
                }
            )

        busy_sessions = [s for s in sessions if s["busy"]]
        return {
            "busy": bool(busy_sessions),
            "busy_count": len(busy_sessions),
            "active_session_id": active,
            "sessions": sessions,
            "safe_to_restart": not busy_sessions,
            "checked_at": now,
        }

    @staticmethod
    def _queue_preview(item: dict) -> str:
        """대기 말풍선에 보여줄 문구."""
        raw = item.get("user_input")
        if isinstance(raw, str):
            if raw.strip():
                return raw
            # 첨부 이미지만 올려 보낸 경우 — 빈 칩이 뜨지 않게 장수로 대신한다.
            uploaded = sum(
                1
                for img in (item.get("images") or [])
                if isinstance(img, dict) and img.get("source") == "upload"
            )
            if uploaded:
                return f"[이미지 {uploaded}장]"
            return raw
        return "[음성 입력]"

    def _enqueue_pending(self, session_id: str, item: dict) -> dict:
        """작업 중인 세션의 입력을 대기 큐에 쌓는다."""
        queue = self._pending_inputs.setdefault(session_id, [])
        while len(queue) >= self._QUEUE_MAX_ITEMS:
            dropped = queue.pop(0)
            logger.warning(
                f"[queue] 대기 입력 상한({self._QUEUE_MAX_ITEMS}) 초과 — 가장 오래된 항목 폐기: "
                f"{self._queue_preview(dropped['item'])[:40]}"
            )
        entry = {"id": f"q_{uuid.uuid4().hex[:8]}", "ts": time.time(), "item": item}
        queue.append(entry)
        logger.info(
            f"[queue] 작업 중이라 입력을 대기열에 넣음 (session={session_id}, "
            f"대기 {len(queue)}건): {self._queue_preview(item)[:40]}"
        )
        return entry

    async def _notify_session_queue(self, session_id: str) -> None:
        """그 세션의 대기 목록 스냅샷을 프론트에 알린다 (빈 배열이면 비었다는 뜻)."""
        items = [
            {"id": e["id"], "text": self._queue_preview(e["item"]), "ts": e["ts"]}
            for e in self._pending_inputs.get(session_id, [])
        ]
        payload = json.dumps(
            {"type": "session-queue", "session_id": session_id, "items": items},
            ensure_ascii=False,
        )
        for ws in list(self.client_connections.values()):
            try:
                await ws.send_text(payload)
            except Exception:
                pass

    @staticmethod
    def _queue_mergeable(entry: dict) -> bool:
        """다른 입력과 한 턴으로 합쳐도 되는 항목인지.

        이미지·시스템 오디오·자발 발화가 붙은 입력은 따로 처리한다
        (첨부가 섞이면 어느 말에 딸린 것인지 알 수 없게 된다).
        """
        item = entry.get("item") or {}
        meta = item.get("metadata") or {}
        return (
            isinstance(item.get("user_input"), str)
            and bool(str(item.get("user_input")).strip())
            and not item.get("images")
            and not meta.get("system_audio")
            and not meta.get("proactive_speak")
        )

    def _merge_pending(self, session_id: str) -> Optional[dict]:
        """다음 턴으로 돌릴 입력 하나를 대기 큐에서 꺼낸다.

        연달아 적은 글은 한 덩어리로 합쳐서 턴 수를 줄이고, 시간 간격이
        벌어진 글이나 첨부가 붙은 입력은 합치지 않고 다음 턴으로 넘긴다.
        """
        queue = self._pending_inputs.get(session_id) or []
        if not queue:
            self._pending_inputs.pop(session_id, None)
            return None
        first = queue.pop(0)
        item = first["item"]
        if self._queue_mergeable(first):
            texts = [str(item.get("user_input", ""))]
            last_ts = first["ts"]
            while (
                queue
                and self._queue_mergeable(queue[0])
                and queue[0]["ts"] - last_ts <= self._QUEUE_MERGE_SLACK
            ):
                nxt = queue.pop(0)
                texts.append(str(nxt["item"].get("user_input", "")))
                last_ts = nxt["ts"]
            if len(texts) > 1:
                item = dict(item)
                item["user_input"] = "\n".join(texts)
                logger.info(
                    f"[queue] 연달아 적은 입력 {len(texts)}건을 한 턴으로 합침 "
                    f"(session={session_id})"
                )
        if not queue:
            self._pending_inputs.pop(session_id, None)
        return item

    async def _drain_pending(self, session_id: str) -> None:
        """턴이 끝난 뒤 대기분을 후속 작업처럼 이어서 돌린다."""
        if self._session_busy(session_id):
            return
        if not self._pending_inputs.get(session_id):
            return
        if session_id in self._skip_drain:
            # 인터럽트로 멈춘 직후. 쌓인 입력은 지우지 않고 보류만 한다.
            self._skip_drain.discard(session_id)
            logger.info(f"[queue] 인터럽트 직후라 자동 처리를 건너뜀 (session={session_id})")
            return
        starter = self._start_turn
        if starter is None:
            logger.warning("[queue] 파이프라인이 없어 대기 입력을 이어 돌리지 못함")
            return
        item = self._merge_pending(session_id)
        if item is None:
            return
        logger.info(
            f"[queue] 대기 입력을 이어서 처리 (session={session_id}, "
            f"남은 대기 {len(self._pending_inputs.get(session_id, []))}건)"
        )
        await starter(item, session_id)
        await self._notify_session_queue(session_id)

    async def _flush_session_buffer(self, session_id: str) -> None:
        """세션으로 넘어왔을 때 보류분을 재생 큐에 되돌린다.

        소리 정책(네무린 확정): 마지막 한 덩어리만 음성으로 재생하고
        그 앞의 턴들은 자막만 표시한다.
        """
        turns = self._session_buffers.pop(session_id, None)
        if not turns:
            return
        # 앞의 턴들은 글(history 복원)로 이미 보이므로 버리고, 마지막 한 덩어리만 음성 재생.
        last = self._mark_replay_only(turns[-1])
        logger.info(
            f"[session] 보류 발화 재생 (session={session_id}, "
            f"{len(turns)}턴 중 마지막 턴만 음성)"
        )
        ready = asyncio.Event()
        ready.set()
        pq = self._regular_pq
        if pq is not None:
            await pq.put((last, ready, session_id))
        else:
            await self._replay_to_client(last)

    async def _replay_to_client(self, messages: List[str]) -> None:
        """파이프라인이 없을 때 쓰는 단순 전송 (순서 지연 없음)."""
        if not self.client_connections:
            return
        ws = self.client_connections.get(next(reversed(self.client_connections)))
        if not ws:
            return
        for raw in messages:
            try:
                await ws.send_text(raw)
            except Exception:
                break

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "fetch-letta-history": self._handle_fetch_letta_history,
            # 다중 세션 (TODO19 2단계)
            "fetch-session-list": self._handle_fetch_session_list,
            "create-session": self._handle_create_session,
            "switch-session": self._handle_switch_session,
            "delete-session": self._handle_delete_session,
            "rename-session": self._handle_rename_session,
            # 다중 세션 동시 처리 (TODO19 3단계)
            "replay-session-buffer": self._handle_replay_session_buffer,
            # 작업 중 입력 큐잉 (TODO25)
            "cancel-queued-input": self._handle_cancel_queued_input,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
            "vtuber-screen-position": self._handle_vtuber_screen_position,
            "tool-approval-response": self._handle_tool_approval_response,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            logger.info(f"Connection established for client {client_uid}")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])

        self.chat_group_manager.client_group_map[client_uid] = ""
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        wind_config = None
        idle_gaze_config = None
        if session_service_context.system_config:
            sc = session_service_context.system_config
            if sc.wind_effect:
                wind_config = sc.wind_effect.model_dump()
            if sc.idle_gaze:
                idle_gaze_config = sc.idle_gaze.model_dump()

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "character_name": session_service_context.character_config.character_name,
                    "human_name": session_service_context.character_config.human_name,
                    "client_uid": client_uid,
                    "wind_effect_config": wind_config,
                    "idle_gaze_config": idle_gaze_config,
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        # Microphone is off by default — user enables it manually via the UI

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            agent_engine=self.default_context_cache.agent_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=send_text,
            client_uid=client_uid,
        )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")
            else:
                # 프론트가 오디오 재생을 끝냈다 = 네무린이 발화를 다 들었다.
                mark_turn_stage("played")

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_disconnect(self, client_uid: str) -> None:
        """Handle client disconnection"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient)
        context = self.client_contexts.get(client_uid)
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        message_handler.cleanup_client(client_uid)

    async def trigger_from_live(self, text: str) -> Optional[asyncio.Task]:
        """Trigger a conversation turn from a live platform event.

        Uses the most recently connected client's WebSocket and context.
        Returns the created asyncio.Task, or None if no client is connected.
        """
        if not self.client_connections:
            logger.debug("No active clients — dropping live event")
            return None
        client_uid = next(reversed(self.client_connections))
        websocket = self.client_connections.get(client_uid)
        context = self.client_contexts.get(client_uid)
        if not websocket or not context:
            return None
        await handle_conversation_trigger(
            msg_type="text-input",
            data={"type": "text-input", "text": text},
            client_uid=client_uid,
            context=context,
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
            initial_metadata={"from_live": True},
        )
        # handle_conversation_trigger stores the task synchronously before returning
        return self.current_conversation_tasks.get(client_uid)

    async def start_live_event_loop(self) -> None:
        """Background loop for live platform events.

        Two-tier pipeline:
        - regular_playback_queue: chats and user input (overload at 10)
        - priority_playback_queue: donations and subscriptions (buffer 100)

        LLM + TTS run in parallel for both tiers. The player always drains
        priority_playback_queue before taking from regular_playback_queue.
        """
        import random
        from .live.live_chat_injector import get_immediate_queue, get_priority_queue

        REGULAR_MAX_BUFFER = 5
        PRIORITY_MAX_BUFFER = 10

        # 오버로드 시 LLM 없이 즉시 TTS로 출력하는 고정 대사
        _OVERLOAD_LINES = [
            "야 잠깐만, 채팅 너무 빨라! 눈이 빙글빙글 돌아간다고!",
            "다들 천천히 좀! 나 AI지만 눈은 두 개밖에 없거든?!",
            "헉, 채팅 폭주다! 좀만 기다려, 지금 머리가 과열 중이야!",
            "와 채팅 속도 미쳤어. 잠깐 숨 좀 돌리자!",
            "다들 손가락이 왜 그렇게 빨라?! 나 읽는 속도가 안 따라가!",
            "아아아 너무 많아! 좀만 천천히, 나도 다 읽고 싶단 말이야!",
            "잠깐 타임! 채팅이 폭포수처럼 쏟아지고 있어!",
            "다들 흥분한 거 알겠는데, 나 지금 눈알 풀 가동 중이거든?!",
            "채팅 속도 왜 이래! 좀만 살살, 내 처리 속도에도 한계가 있다고!",
        ]

        # Ordered playback queues:
        # each slot is (messages: list, ready: Event, session_id: str|None)
        regular_pq: asyncio.Queue = asyncio.Queue()
        priority_pq: asyncio.Queue = asyncio.Queue()
        # 비활성 세션 버퍼를 되돌릴 때 쓰도록 핸들러에 노출
        self._regular_pq = regular_pq

        # ── helpers ──────────────────────────────────────────────────────────

        async def _capture(
            item: dict,
            messages: list,
            ready: asyncio.Event,
            session_id: Optional[str] = None,
        ) -> None:
            """Run LLM+TTS in capture mode; fill messages, then set ready.

            session_id 를 이 task 의 ContextVar 에 고정해서, 다른 세션이 동시에
            돌아도 대화 저장 위치와 IPC 파일 경로가 중간에 갈아타지 않게 한다.
            """
            from .agent.agents.letta_agent import bind_session, progress_notify_file

            sid = str(session_id or self._active_session_id() or "default")
            bind_session(sid)
            # 세션별 task 추적 — 인터럽트 시 cancel 대상
            self._capture_tasks[sid] = asyncio.current_task()
            self._turn_started[sid] = time.time()
            user_input_preview = str(item.get("user_input", ""))[:40]
            logger.debug(f"[TTS-DBG] capture start: {user_input_preview!r}")
            try:
                if not self.client_connections:
                    return
                client_uid = next(reversed(self.client_connections))
                ws = self.client_connections.get(client_uid)
                context = self.client_contexts.get(client_uid)
                if not context:
                    return

                user_input = item.get("user_input", "")
                images = item.get("images")
                base_meta = dict(item.get("metadata") or {})
                base_meta.pop("session_id", None)
                base_meta["skip_playback_wait"] = True

                captured: list[str] = []
                _seen_notify_ts: set = set()
                _sent_progress_texts: set = set()

                # [알림: ...] 중간 진행 알림을 캡처 버퍼 우회해 실시간 전송
                import os as _os_notify
                # 다중 세션: 알림 파일은 letta conversation 단위로 분리되어 있다.
                # 이 capture 가 담당한 세션 경로를 직접 계산한다(활성 세션이 바뀌어도 불변).
                _NOTIFY_FILE = progress_notify_file(sid)

                async def _poll_and_send_notify() -> None:
                    while True:
                        try:
                            if ws and _os_notify.path.exists(_NOTIFY_FILE):
                                with open(_NOTIFY_FILE) as _nf:
                                    for _line in _nf:
                                        _line = _line.strip()
                                        if not _line:
                                            continue
                                        try:
                                            _entry = json.loads(_line)
                                            _ts = _entry.get("ts")
                                            if _ts and _ts not in _seen_notify_ts:
                                                _seen_notify_ts.add(_ts)
                                                _text = _entry.get("text", "")
                                                if not _text or _text in _sent_progress_texts:
                                                    continue
                                                _sent_progress_texts.add(_text)
                                                # letta 에는 알림이 저장되지 않으므로
                                                # 세션 보조 로그에 남겨 복원에 쓴다.
                                                session_log.log_progress(
                                                    sid, _text, _ts
                                                )
                                                await ws.send_text(json.dumps({
                                                    "type": "progress_update",
                                                    "text": _text,
                                                    "timestamp": _ts,
                                                    "session_id": sid,
                                                }))
                                        except Exception:
                                            pass
                        except Exception:
                            pass
                        await asyncio.sleep(0.3)

                poll_task = asyncio.create_task(_poll_and_send_notify())

                async def _cap_send(raw: str) -> None:
                    # image_display는 오디오 순서와 무관하므로 즉시 전송 → captured 제외
                    try:
                        _payload = json.loads(raw)
                        _msg_type = _payload.get("type")
                        if _msg_type == "progress_update":
                            _text = _payload.get("text", "")
                            if ws and _text and _text not in _sent_progress_texts:
                                _sent_progress_texts.add(_text)
                                _ts = _payload.get("timestamp") or time.time()
                                session_log.log_progress(sid, _text, _ts)
                                _payload["session_id"] = sid
                                await ws.send_text(json.dumps(_payload))
                            return
                        if _msg_type == "image_display":
                            if ws:
                                try:
                                    await ws.send_text(raw)
                                    return
                                except Exception as _img_err:
                                    logger.debug(
                                        f"[TTS-DBG] image_display 실시간 전송 실패, capture fallback: {_img_err}"
                                    )
                            captured.append(raw)
                            return
                    except Exception:
                        pass
                    captured.append(raw)

                try:
                    await process_single_conversation(
                        context=context,
                        websocket_send=_cap_send,
                        client_uid=client_uid,
                        user_input=user_input,
                        images=images,
                        metadata=base_meta,
                    )
                finally:
                    poll_task.cancel()
                    try:
                        await poll_task
                    except asyncio.CancelledError:
                        pass

                messages.extend(captured)
                # Count captured audio chunks for diagnostics
                real_audio = sum(
                    1 for m in captured
                    if "\"type\": \"audio\"" in m and "\"audio\": null" not in m
                    and "\"audio\": None" not in m
                )
                logger.debug(
                    f"[TTS-DBG] capture done: {user_input_preview!r} "
                    f"({len(captured)} msgs, {real_audio} audio chunks)"
                )
            except asyncio.CancelledError:
                ready.set()
                raise
            except Exception as e:
                logger.error(f"Live capture error: {e}")
            finally:
                # task 추적 해제 (이 task가 현재 추적 중인 것과 같을 때만)
                if self._capture_tasks.get(sid) is asyncio.current_task():
                    self._capture_tasks.pop(sid, None)
                ready.set()
                # 작업 중이라 대기열에 쌓인 입력이 있으면 곧바로 이어서 돌린다 (TODO25).
                # 이 task 안에서 await 하면 자기 자신이 끝나지 않으므로 별 task 로 띄운다.
                if self._pending_inputs.get(sid):
                    asyncio.create_task(self._drain_pending(sid))

        async def _capture_direct_tts(text: str, messages: list, ready: asyncio.Event) -> None:
            """LLM 없이 TTS를 직접 생성해 messages에 채운다 (오버로드 즉시 응답용)."""
            try:
                if not self.client_connections:
                    return
                client_uid = next(reversed(self.client_connections))
                context = self.client_contexts.get(client_uid)
                if not context:
                    return

                from .conversations.tts_manager import TTSTaskManager
                from .agent.output_types import DisplayText, Actions

                captured: list[str] = []

                async def _cap_send(raw: str) -> None:
                    captured.append(raw)

                await _cap_send(json.dumps({"type": "control", "text": "conversation-chain-start"}))
                await _cap_send(json.dumps({"type": "full-text", "text": text}))

                tts_manager = TTSTaskManager()
                display = DisplayText(
                    text=text,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )
                await tts_manager.speak(
                    tts_text=text,
                    display_text=display,
                    actions=Actions(),
                    live2d_model=context.live2d_model,
                    tts_engine=context.tts_engine,
                    websocket_send=_cap_send,
                )
                if tts_manager.task_list:
                    await asyncio.gather(*tts_manager.task_list)
                    await tts_manager.wait_for_delivery()
                    await _cap_send(json.dumps({"type": "backend-synth-complete"}))

                await _cap_send(json.dumps({"type": "force-new-message"}))
                await _cap_send(json.dumps({"type": "control", "text": "conversation-chain-end"}))
                messages.extend(captured)
            except asyncio.CancelledError:
                ready.set()
                raise
            except Exception as e:
                logger.error(f"Direct TTS overload error: {e}")
            finally:
                ready.set()

        async def _notify_llm_skipped(drained_count: int) -> None:
            """채팅 스킵 사실을 Letta 메모리에 조용히 기록한다 (TTS 없음)."""
            try:
                if not self.client_connections:
                    return
                client_uid = next(reversed(self.client_connections))
                context = self.client_contexts.get(client_uid)
                if not context:
                    return
                agent = context.agent_engine
                if not (hasattr(agent, "client") and hasattr(agent, "id")):
                    return
                notification = (
                    f"[시스템] 채팅 폭주로 인해 {drained_count}개의 채팅이 스킵되어 읽지 못했습니다."
                )
                await agent.client.agents.messages.create(
                    agent_id=agent.id,
                    streaming=False,  # type: ignore[call-overload]
                    input=notification,
                )
                logger.debug(f"[Overload] Letta에 {drained_count}개 스킵 알림 완료")
            except Exception as e:
                logger.debug(f"[Overload] Letta 알림 실패 (무시): {e}")

        async def _replay(
            messages: list, idx: int = 0, session_id: Optional[str] = None
        ) -> None:
            """Send captured messages with progressive audio delivery.

            In normal (non-capture) mode audio chunks arrive one at a time as TTS
            generates (~2-3s per sentence). In replay mode all chunks would arrive
            within milliseconds, causing the frontend to play them simultaneously.
            Progressive delivery adds a per-chunk delay equal to its duration so the
            frontend receives chunks at the same pace as normal mode.
            """
            if not messages or not self.client_connections:
                return
            # 비활성 세션의 발화는 지금 보여주지 않고 모아둔다.
            # 사용자가 그 세션으로 넘어올 때 꺼내서 재생한다(마지막 턴만 음성).
            if session_id and session_id != self._active_session_id():
                self._buffer_session_turn(session_id, list(messages))
                await self._notify_session_activity(session_id)
                return
            client_uid = next(reversed(self.client_connections))
            ws = self.client_connections.get(client_uid)
            if not ws:
                return

            total_audio_ms = 0
            audio_chunk_count = 0
            send_failed = False
            # 재생 도중 사용자가 다른 세션으로 넘어가면 남은 발화를 여기에 담아
            # 그 세션 보류 버퍼로 되돌린다. 돌아오면 멈춘 자리부터 이어서 말한다.
            leftover: Optional[list] = None

            def _left_session() -> bool:
                """사용자가 이 세션을 떠났는지."""
                return bool(session_id) and session_id != self._active_session_id()

            async def _play_wait(duration: float) -> bool:
                """오디오 한 덩어리가 재생되는 동안 대기. 중간에 세션이 바뀌면 True.

                통째로 sleep 하지 않고 잘게 끊어 기다린다. 그래야 떠난 걸 빨리 알아채고
                뒤 문장을 화면(=남의 세션)으로 흘려보내지 않는다.
                """
                waited = 0.0
                step = 0.2
                while waited < duration:
                    await asyncio.sleep(min(step, duration - waited))
                    waited += step
                    if _left_session():
                        return True
                return False

            if session_id:
                self._live_replays.add(session_id)
            try:
                for pos, raw in enumerate(messages):
                    # 앞 문장 재생이 끝난 직후 떠났다면, 그 다음 문장부터 보류한다.
                    if _left_session():
                        leftover = list(messages[pos:])
                        break
                    await ws.send_text(raw)
                    try:
                        data = json.loads(raw)
                        if data.get("type") == "audio" and data.get("audio"):
                            volumes = data.get("volumes") or []
                            slice_ms = data.get("slice_length") or 20
                            chunk_ms = len(volumes) * slice_ms
                            total_audio_ms += chunk_ms
                            audio_chunk_count += 1
                            logger.debug(
                                f"[TTS-DBG #{idx}]   audio chunk {audio_chunk_count}: "
                                f"{chunk_ms}ms  (cumul={total_audio_ms}ms)"
                            )
                            # Progressive: wait for this chunk to play before sending next
                            if await _play_wait(chunk_ms / 1000):
                                # 이 문장을 말하던 중에 떠났다 → 프론트가 재생을 끊었으므로
                                # 반만 들린 이 문장부터 다시 들려준다.
                                leftover = list(messages[pos:])
                                break
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[TTS-DBG #{idx}] Error during replay send: {e}")
                send_failed = True
            finally:
                if session_id:
                    self._live_replays.discard(session_id)
                if leftover:
                    self._buffer_session_turn(str(session_id), leftover, front=True)
                    logger.info(
                        f"[TTS-DBG #{idx}] ⏸ 세션 이동으로 재생 중단 "
                        f"(session={session_id}, 남은 {len(leftover)}개 보류)"
                    )
                    await self._notify_session_activity(str(session_id))
                elif total_audio_ms > 0:
                    if send_failed:
                        await asyncio.sleep(2.0)
                    else:
                        # Small buffer after the last chunk finishes
                        logger.info(
                            f"[TTS-DBG #{idx}] ⏳ progressive done: "
                            f"{audio_chunk_count} chunks, {total_audio_ms}ms"
                        )
                        await asyncio.sleep(0.3)
                elif audio_chunk_count == 0:
                    logger.info(f"[TTS-DBG #{idx}]   no real audio — skipping wait")

        # ── player ───────────────────────────────────────────────────────────

        _play_idx = 0  # monotonic counter to correlate log lines per output

        async def _player() -> None:
            """먼저 완성된 슬롯을 먼저 재생한다 (completion-order playback)."""
            import time
            nonlocal _play_idx
            SLOT_TIMEOUT = 7200.0  # 슬롯당 최대 대기 시간 (초) — 2시간
            # pending: (messages, ready_event, session_id, enqueued_at)
            pending: list[tuple[list, asyncio.Event, Optional[str], float]] = []

            def _drain_queues() -> None:
                """큐에서 즉시 꺼낼 수 있는 슬롯을 모두 pending에 추가."""
                while True:
                    try:
                        msgs, rdy, sid = priority_pq.get_nowait()
                        pending.insert(0, (msgs, rdy, sid, time.monotonic()))
                    except asyncio.QueueEmpty:
                        break
                while True:
                    try:
                        msgs, rdy, sid = regular_pq.get_nowait()
                        pending.append((msgs, rdy, sid, time.monotonic()))
                    except asyncio.QueueEmpty:
                        break

            def _purge_timed_out() -> None:
                """타임아웃이 지난 슬롯을 버린다."""
                now = time.monotonic()
                for i in reversed(range(len(pending))):
                    _, _, slot_sid, enqueued_at = pending[i]
                    if now - enqueued_at > SLOT_TIMEOUT:
                        pending.pop(i)
                        logger.warning("[TTS-DBG] ⏰ slot timed out (LLM/TTS >7200s) — skipping")
                        # slot timeout 시 그 세션의 claude -p 프로세스만 종료
                        try:
                            if self.client_connections:
                                _uid = next(reversed(self.client_connections))
                                _ctx = self.client_contexts.get(_uid)
                                if _ctx and hasattr(_ctx, "agent_engine"):
                                    _agent = _ctx.agent_engine
                                    if hasattr(_agent, "handle_interrupt"):
                                        try:
                                            _agent.handle_interrupt(
                                                "", conversation_id=slot_sid
                                            )
                                        except TypeError:
                                            _agent.handle_interrupt("")
                                        logger.info(
                                            "[TTS-DBG] slot timeout → "
                                            f"handle_interrupt(session={slot_sid}) 호출"
                                        )
                        except Exception as _e:
                            logger.warning(f"[TTS-DBG] slot timeout handle_interrupt 실패 (무시): {_e}")

            while True:
                _drain_queues()
                _purge_timed_out()

                # ── pending이 비어있으면 새 슬롯이 올 때까지 블록 ─────────────────
                if not pending:
                    get_prio = asyncio.create_task(priority_pq.get())
                    get_reg  = asyncio.create_task(regular_pq.get())
                    done, _ = await asyncio.wait(
                        {get_prio, get_reg}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in {get_prio, get_reg} - done:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
                    now = time.monotonic()
                    if get_prio in done:
                        msgs, rdy, sid = get_prio.result()
                        pending.insert(0, (msgs, rdy, sid, now))
                        if get_reg in done:
                            msgs, rdy, sid = get_reg.result()
                            pending.append((msgs, rdy, sid, now))
                    else:
                        msgs, rdy, sid = get_reg.result()
                        pending.append((msgs, rdy, sid, now))
                    continue

                # ── 이미 완성된 슬롯이 있는지 즉시 확인 ─────────────────────────
                ready_idx = next(
                    (i for i, (_, rdy, _s, _t) in enumerate(pending) if rdy.is_set()),
                    None,
                )

                if ready_idx is None:
                    # 타임아웃까지 남은 시간 계산 (가장 오래된 슬롯 기준)
                    now = time.monotonic()
                    wait_timeout = max(
                        0.1,
                        SLOT_TIMEOUT - max(now - t for _, _, _s, t in pending)
                    )

                    # pending 은 대기 중에도 새 슬롯이 끼어들 수 있으므로
                    # 이 시점의 순서를 그대로 떠서 wait task 와 짝을 맞춘다.
                    snapshot = list(pending)
                    ready_waits = [
                        asyncio.create_task(rdy.wait()) for _, rdy, _s, _t in snapshot
                    ]
                    get_prio    = asyncio.create_task(priority_pq.get())
                    get_reg     = asyncio.create_task(regular_pq.get())

                    all_tasks = ready_waits + [get_prio, get_reg]
                    done, _ = await asyncio.wait(
                        all_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=wait_timeout,
                    )

                    for t in all_tasks:
                        if t not in done:
                            t.cancel()
                            try:
                                await t
                            except (asyncio.CancelledError, Exception):
                                pass

                    # 새로 도착한 슬롯 추가
                    now = time.monotonic()
                    if get_prio in done:
                        try:
                            msgs, rdy, sid = get_prio.result()
                            pending.insert(0, (msgs, rdy, sid, now))
                        except Exception:
                            pass
                    if get_reg in done:
                        try:
                            msgs, rdy, sid = get_reg.result()
                            pending.append((msgs, rdy, sid, now))
                        except Exception:
                            pass

                    snap_idx = next(
                        (i for i, t in enumerate(ready_waits) if t in done), None
                    )
                    if snap_idx is None:
                        continue  # 새 슬롯 도착 또는 타임아웃 체크, 다시 루프
                    # 스냅샷에서 고른 슬롯을 현재 pending 에서 동일 객체로 제거
                    target_slot = snapshot[snap_idx]
                    ready_idx = next(
                        (i for i, p in enumerate(pending) if p is target_slot), None
                    )
                    if ready_idx is None:
                        continue  # 이미 타임아웃으로 버려진 슬롯

                # ── 완성된 슬롯 재생 ──────────────────────────────────────────────
                msgs, _, slot_session_id, _ = pending.pop(ready_idx)
                _play_idx += 1
                idx = _play_idx
                logger.info(
                    f"[TTS-DBG #{idx}] ▶ replay start  "
                    f"(pq_regular={regular_pq.qsize()} pq_priority={priority_pq.qsize()} pending={len(pending)})"
                )
                try:
                    await _replay(msgs, idx, slot_session_id)
                    logger.info(f"[TTS-DBG #{idx}] ✓ replay done")
                    mark_turn_stage("delivered")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Live playback error: {e}")

        # ── 턴 시작 (일반 입력 / 대기열 소화 공용, TODO25) ────────────────────

        async def _start_turn(item: dict, session_id: str) -> None:
            """입력 하나를 실제 턴으로 올린다.

            capture task 등록을 여기서 바로 해두는 게 중요하다. _capture 안에서만
            등록하면 task 가 처음 깨어나기 전까지 "작업 중"으로 보이지 않아서,
            그 틈에 들어온 입력이 큐를 거치지 않고 끼어들 수 있다.
            """
            meta = item.get("metadata") or {}
            if meta.pop("_log_user_text", False):
                text = item.get("user_input")
                if isinstance(text, str) and text.strip():
                    session_log.log_user_message(session_id, text)
            msgs: list = []
            rdy = asyncio.Event()
            await regular_pq.put((msgs, rdy, session_id))
            task = asyncio.create_task(_capture(item, msgs, rdy, session_id))
            self._capture_tasks[session_id] = task

        # 턴이 끝난 뒤 대기열을 소화할 때 쓰도록 핸들러에 노출
        self._start_turn = _start_turn

        # ── priority consumer (donations, subscriptions) ──────────────────────

        async def _consume_priority() -> None:
            pq = get_priority_queue()
            while True:
                item = await pq.get()
                # Drop oldest items if playback buffer is full
                if priority_pq.qsize() >= PRIORITY_MAX_BUFFER:
                    logger.warning(
                        f"Priority playback buffer full ({PRIORITY_MAX_BUFFER}), dropping oldest"
                    )
                    try:
                        priority_pq.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                messages: list = []
                ready = asyncio.Event()
                sid = (item.get("metadata") or {}).get(
                    "session_id"
                ) or self._active_session_id()
                await priority_pq.put((messages, ready, sid))
                asyncio.create_task(_capture(item, messages, ready, sid))

        # ── regular consumer (chats, user input) ─────────────────────────────

        async def _consume_regular() -> None:
            rq = get_immediate_queue()

            CHAT_BUFFER_THRESHOLD = 3    # Start buffering live chats when pq depth > this
            CHAT_BUFFER_MAX_SIZE = 5     # Flush buffer when this many chats accumulate
            CHAT_BUFFER_TIMEOUT = 5.0    # Flush buffer after this many seconds
            BATCH_HEADER = "[시청자 채팅]"

            chat_buffer: list[str] = []  # Accumulates live chat texts during buffer mode
            _buffer_timer: asyncio.Task | None = None

            async def _flush_chat_buffer() -> None:
                nonlocal _buffer_timer
                if _buffer_timer and not _buffer_timer.done():
                    _buffer_timer.cancel()
                _buffer_timer = None
                if not chat_buffer:
                    return
                texts = list(chat_buffer)
                chat_buffer.clear()
                batch_text = BATCH_HEADER + "\n" + "\n".join(texts)
                logger.info(f"[Chat Buffer Flush] {len(texts)}개 채팅 배치 전송")
                batch_item = {
                    "user_input": batch_text,
                    "images": None,
                    "metadata": {"from_live": True},
                }
                msgs, rdy = [], asyncio.Event()
                sid = self._active_session_id()
                await regular_pq.put((msgs, rdy, sid))
                asyncio.create_task(_capture(batch_item, msgs, rdy, sid))

            async def _buffer_timeout() -> None:
                await asyncio.sleep(CHAT_BUFFER_TIMEOUT)
                if chat_buffer:
                    logger.debug("[Chat Buffer] 5초 타임아웃, flush")
                    await _flush_chat_buffer()

            while True:
                item = await rq.get()

                # Incoming overload check (too many items in the incoming queue)
                pending = rq.qsize()
                if pending > REGULAR_MAX_BUFFER:
                    drained = 0
                    while not rq.empty():
                        try:
                            rq.get_nowait()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    await _flush_chat_buffer()
                    overload_line = random.choice(_OVERLOAD_LINES)
                    logger.warning(
                        f"Regular queue overloaded ({pending} pending, drained {drained})"
                        f" — direct TTS: {overload_line[:40]}"
                    )
                    msgs, rdy = [], asyncio.Event()
                    await regular_pq.put((msgs, rdy, self._active_session_id()))
                    asyncio.create_task(_capture_direct_tts(overload_line, msgs, rdy))
                    asyncio.create_task(_notify_llm_skipped(drained))
                    continue

                # Determine if this is a live chat eligible for buffering
                is_live_chat = (
                    isinstance(item.get("user_input"), str)
                    and bool(item.get("metadata", {}).get("from_live"))
                )

                if is_live_chat and regular_pq.qsize() > CHAT_BUFFER_THRESHOLD:
                    # Playback queue is deep — buffer this chat instead of
                    # immediately starting a new LLM call
                    is_first = len(chat_buffer) == 0
                    chat_buffer.append(item["user_input"])
                    logger.debug(
                        f"[Chat Buffer] pq={regular_pq.qsize()} "
                        f"→ buffered ({len(chat_buffer)}개): {item['user_input'][:60]}"
                    )
                    if is_first:
                        _buffer_timer = asyncio.create_task(_buffer_timeout())
                    if len(chat_buffer) >= CHAT_BUFFER_MAX_SIZE:
                        logger.debug("[Chat Buffer] 최대 크기 도달, flush")
                        await _flush_chat_buffer()
                    continue

                # Normal path: flush any pending buffer first so it plays
                # before (or merged into) the current item
                if chat_buffer:
                    await _flush_chat_buffer()

                # 입력이 들어온 시점의 세션을 꼬리표로 달아둔다. 이후 사용자가
                # 다른 세션으로 옮겨가도 이 작업은 자기 세션에 그대로 저장된다.
                sid = (item.get("metadata") or {}).get(
                    "session_id"
                ) or self._active_session_id()

                # 그 세션이 이미 턴을 돌고 있으면 끼어들지 않는다 (TODO25).
                # 같은 대화 흐름에 LLM 호출을 겹쳐 넣으면 응답이 섞이므로,
                # 대기열에 쌓아두고 지금 턴이 끝나는 즉시 이어서 돌린다.
                if self._session_busy(sid):
                    self._enqueue_pending(sid, item)
                    await self._notify_session_queue(sid)
                    continue

                await _start_turn(item, sid)

        # ── run all three concurrently ────────────────────────────────────────

        tasks = [
            asyncio.create_task(_player()),
            asyncio.create_task(_consume_priority()),
            asyncio.create_task(_consume_regular()),
        ]
        logger.info(
            "Live event pipeline started "
            "(parallel LLM+TTS, priority playback for donations/subscriptions)"
        )
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            logger.info("Live event consumer loop stopped")

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption.

        1. 현재 실행 중인 capture task (LLM+TTS) 취소
        2. LettaAgent의 abort 플래그 설정 → 스트리밍 루프 조기 종료
        3. 프론트엔드에 conversation-chain-end 전송 → aiState → idle
        """
        logger.info(f"[Interrupt] {client_uid}에서 인터럽트 신호 수신")

        # 1. 지금 보고 있는 세션의 capture task만 취소 (다른 세션 작업은 보존)
        target_sid = str(
            (data.get("session_id") if isinstance(data, dict) else None)
            or self._active_session_id()
        )
        # 인터럽트는 "멈춰"라는 신호다. 대기열에 쌓인 입력을 곧바로 이어 돌리면
        # 멈춘 것처럼 보이지 않으므로, 이번 턴 종료 시 자동 소화를 한 번 건너뛴다.
        # 다만 적어둔 말을 지우지는 않고 보류 상태로 남겨둔다 (TODO25).
        if self._pending_inputs.get(target_sid):
            self._skip_drain.add(target_sid)
            logger.info(
                f"[Interrupt] 대기 입력 {len(self._pending_inputs[target_sid])}건은 "
                f"자동 처리하지 않고 보류 (session={target_sid})"
            )

        task = self._capture_tasks.get(target_sid)
        if task and not task.done():
            logger.info(f"[Interrupt] capture task 취소 (session={target_sid})")
            task.cancel()
        else:
            logger.info(f"[Interrupt] 취소할 capture task 없음 (session={target_sid})")

        # 2. letta agent abort 플래그 설정 (해당 세션만)
        context = self.client_contexts.get(client_uid)
        if context and hasattr(context, "agent_engine"):
            agent = context.agent_engine
            if hasattr(agent, "handle_interrupt"):
                try:
                    agent.handle_interrupt("", conversation_id=target_sid)
                except TypeError:
                    agent.handle_interrupt("")
                logger.info(
                    f"[Interrupt] agent.handle_interrupt(session={target_sid}) 완료"
                )

        # 3. 프론트엔드에 conversation-chain-end 전송
        try:
            await websocket.send_text(
                json.dumps({"type": "control", "text": "conversation-chain-end"})
            )
            logger.info("[Interrupt] conversation-chain-end 전송 완료")
            # 4. 인터럽트 성공 알림 전송 → 프론트에서 자막으로 표시
            await websocket.send_text(
                json.dumps({"type": "control", "text": "interrupt-success"})
            )
            logger.info("[Interrupt] interrupt-success 전송 완료")
        except Exception as e:
            logger.warning(f"[Interrupt] 전송 실패: {e}")

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )

    # ------------------------------------------------------------------
    # 다중 세션(letta conversation) — TODO19 2단계
    # ------------------------------------------------------------------

    def _get_letta_agent(self, client_uid: str) -> Optional[Any]:
        """세션 기능을 지원하는 letta agent 가져오기. 미지원이면 None."""
        context = self.client_contexts.get(client_uid)
        if context is None:
            return None
        agent = getattr(context, "agent_engine", None)
        if agent is None or not hasattr(agent, "list_sessions"):
            return None
        return agent

    async def _send_session_list(
        self, websocket: WebSocket, agent: Any, extra: Optional[dict] = None
    ) -> None:
        try:
            sessions = await agent.list_sessions()
        except Exception as e:
            logger.error(f"[session] 목록 조회 중 오류: {e}")
            sessions = []
        active_id = getattr(agent, "active_conversation_id", None) or getattr(
            agent, "conversation_id", "default"
        )
        payload = {
            "type": "session-list",
            "sessions": sessions,
            "active_id": active_id,
            "supported": True,
            # 다중 세션 동시 처리: 지금 돌고 있는 세션과 읽지 않은 발화가 쌓인 세션
            "running_ids": [sid for sid, t in self._capture_tasks.items() if not t.done()],
            "unread_ids": [sid for sid, turns in self._session_buffers.items() if turns],
            # 세션별 미확인 발화 덩어리 개수 (새로고침 후에도 배지 숫자가 유지된다)
            "unread_counts": {
                sid: len(turns) for sid, turns in self._session_buffers.items() if turns
            },
            # 작업 중이라 아직 전달되지 않고 대기 중인 입력 건수 (TODO25)
            "queued_counts": {
                sid: len(q) for sid, q in self._pending_inputs.items() if q
            },
        }
        if extra:
            payload.update(extra)
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    async def _handle_fetch_session_list(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """세션 목록 요청. letta 이외 agent면 supported=False로 조용히 폴백."""
        agent = self._get_letta_agent(client_uid)
        if agent is None:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "session-list",
                        "sessions": [],
                        "active_id": "default",
                        "supported": False,
                    }
                )
            )
            return
        await self._send_session_list(websocket, agent)
        # 새로고침 후에도 대기 중인 입력 말풍선이 되살아나게 스냅샷을 같이 보낸다
        await self._notify_session_queue(self._active_session_id())

    async def _handle_create_session(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """새 세션 생성 후 곧바로 그 세션으로 전환하고 목록을 갱신해 보낸다."""
        agent = self._get_letta_agent(client_uid)
        if agent is None:
            return
        created = await agent.create_session(summary=data.get("summary") or "")
        if created is None:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "session-error",
                        "action": "create",
                        "message": "세션 생성에 실패했어.",
                    },
                    ensure_ascii=False,
                )
            )
            return
        agent.set_conversation(created["id"], name=created.get("name") or "")
        await self._send_session_list(
            websocket, agent, extra={"created_id": created["id"]}
        )

    async def _handle_switch_session(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """활성 세션 전환. 대화 복원은 프론트가 fetch-letta-history로 따로 요청한다."""
        agent = self._get_letta_agent(client_uid)
        if agent is None:
            return
        target = data.get("conversation_id") or "default"
        # 프론트가 세션 이름을 같이 보내면 바로 반영(없으면 agent가 letta에서 조회).
        active = agent.set_conversation(target, name=data.get("name"))
        pending_turns = len(self._session_buffers.get(active, []))
        still_running = active in self._capture_tasks
        await websocket.send_text(
            json.dumps(
                {
                    "type": "session-switched",
                    "active_id": active,
                    "pending_turns": pending_turns,
                    "running": still_running,
                },
                ensure_ascii=False,
            )
        )
        # 옮겨간 세션에 대기 중인 입력이 있으면 그 목록도 같이 보낸다 (TODO25)
        await self._notify_session_queue(active)

    async def _handle_replay_session_buffer(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """세션으로 넘어와 기록 복원을 끝낸 뒤, 보류된 마지막 발화를 음성으로 재생."""
        target = str(data.get("conversation_id") or self._active_session_id())
        if target != self._active_session_id():
            # 그 사이 또 옮겨갔다면 재생하지 않고 버퍼를 그대로 둔다.
            return
        # 떠난 세션의 재생이 아직 남은 조각을 보류 버퍼로 되돌리는 중일 수 있다.
        # 그걸 기다리지 않으면 빈 버퍼를 보고 지나쳐서, 이어 말하기가 한 번 밀린다.
        for _ in range(20):  # 최대 2초
            if not self._live_replays:
                break
            await asyncio.sleep(0.1)
        await self._flush_session_buffer(target)

    async def _handle_cancel_queued_input(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """대기 중인 입력 한 건 취소 (적어놓고 마음이 바뀐 경우, TODO25)."""
        sid = str(data.get("session_id") or self._active_session_id())
        target = str(data.get("id") or "")
        queue = self._pending_inputs.get(sid) or []
        remain = [e for e in queue if e["id"] != target]
        if len(remain) != len(queue):
            logger.info(f"[queue] 대기 입력 취소 (session={sid}, id={target})")
        if remain:
            self._pending_inputs[sid] = remain
        else:
            self._pending_inputs.pop(sid, None)
        await self._notify_session_queue(sid)

    async def _handle_delete_session(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        agent = self._get_letta_agent(client_uid)
        if agent is None:
            return
        target = data.get("conversation_id") or ""
        # 지워지는 세션에서 돌던 작업과 쌓인 발화도 같이 정리
        _task = self._capture_tasks.pop(target, None)
        if _task and not _task.done():
            _task.cancel()
        self._session_buffers.pop(target, None)
        self._pending_inputs.pop(target, None)
        self._skip_drain.discard(target)
        session_log.drop_session_log(target)
        success = await agent.delete_session(target)
        await self._send_session_list(
            websocket,
            agent,
            extra={"deleted_id": target if success else None},
        )

    async def _handle_rename_session(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        agent = self._get_letta_agent(client_uid)
        if agent is None:
            return
        target = data.get("conversation_id") or ""
        summary = data.get("summary") or ""
        await agent.update_session_summary(target, summary)
        await self._send_session_list(websocket, agent)

    async def _handle_fetch_letta_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """재접속 시 letta conversation에서 직접 지난 대화를 읽어 돌려준다 (TODO19 1단계).

        백엔드 chat_history JSON은 접속마다 새 history_uid로 조각나 쓸모가 없어서,
        대화 원본인 letta를 복원 소스로 쓴다.

        요청 payload:
            limit  : 가져올 개수 (기본 50)
            before : 이 message id보다 과거만 (위로 스크롤 추가 로드)
            conversation_id : 조회할 세션 (생략 시 현재 세션)
        """
        context = self.client_contexts.get(client_uid)
        if context is None:
            return

        agent = getattr(context, "agent_engine", None)
        fetcher = getattr(agent, "fetch_history_messages", None)
        if fetcher is None:
            # letta 이외 agent에서는 조용히 빈 결과 (프론트가 기존 경로로 폴백)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "letta-history-data",
                        "messages": [],
                        "oldest_id": None,
                        "has_more": False,
                        "append": bool(data.get("before")),
                        "supported": False,
                    }
                )
            )
            return

        try:
            result = await fetcher(
                limit=int(data.get("limit") or 50),
                before=data.get("before"),
                conversation_id=data.get("conversation_id"),
            )
        except Exception as e:
            logger.error(f"[history] letta 히스토리 조회 중 오류: {e}")
            result = {"messages": [], "oldest_id": None, "has_more": False}

        restored = result.get("messages", [])
        # 아직 letta 에 저장되지 않은 발화와 중간 알림을 보조 로그에서 덧댄다.
        # 위로 스크롤한 추가 로드(before)에서는 과거 구간만 보내므로 건드리지 않는다.
        if not data.get("before"):
            try:
                restored = session_log.merge_history(
                    data.get("conversation_id") or self._active_session_id(),
                    restored,
                )
            except Exception as e:
                logger.error(f"[history] 보조 로그 병합 실패: {e}")

        await websocket.send_text(
            json.dumps(
                {
                    "type": "letta-history-data",
                    "messages": restored,
                    "oldest_id": result.get("oldest_id"),
                    "has_more": result.get("has_more", False),
                    "append": bool(data.get("before")),
                    "supported": True,
                    "conversation_id": data.get("conversation_id")
                    or self._active_session_id(),
                },
                ensure_ascii=False,
            )
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                        # 접속 시 자동 생성인지(True) 사용자가 새 세션을 만든 건지(False).
                        # 프론트는 자동 생성이면 채팅창을 비우지 않는다 (TODO19 1단계).
                        "initial": bool(data.get("initial")),
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    pass  # Interrupt disabled — only the explicit button may interrupt
                elif audio_bytes == b"<|RESUME|>":
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )

    async def _handle_conversation_trigger(
        self, _websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Route a conversation trigger into the unified live event queue.

        All inputs — user speech, user text, proactive speak, and live platform
        events — share the same ordered pipeline so nothing interrupts playback.
        """
        from .live.live_chat_injector import put_immediate

        msg_type = data.get("type", "")
        context = self.client_contexts.get(client_uid)

        if msg_type == "ai-speak-signal":
            mode = data.get("mode", "broadcast")
            if mode not in ("broadcast", "private"):
                mode = "broadcast"
            try:
                from .conversations.conversation_handler import _build_proactive_user_input
                user_input = _build_proactive_user_input(context, client_uid, mode=mode)
            except Exception as e:
                logger.error(f"Proactive 입력 생성 실패: {e}")
                user_input = "[PROACTIVE_SPEAK_TRIGGER]\n뭔가 자연스럽게 혼잣말을 해줘.\n[/PROACTIVE_SPEAK_TRIGGER]"
            # 프론트 사이드바에서 입력한 지시문을 기본 프롬프트 뒤에 덧붙인다
            instruction = data.get("proactive_instruction")
            if isinstance(instruction, str) and instruction.strip():
                user_input = (
                    f"{user_input}\n"
                    f"[PROACTIVE_USER_INSTRUCTION]\n"
                    f"{instruction.strip()}\n"
                    f"[/PROACTIVE_USER_INSTRUCTION]"
                )
                logger.info(f"[Proactive] 프론트 지시문 적용: {instruction.strip()[:60]}")
            metadata = {
                "proactive_speak": True,
                "proactive_mode": mode,
                "skip_memory": True,
                "skip_history": True,
            }
            # private 모드: 화면+오디오 입력 같이 받기 (broadcast는 기존대로 images=None)
            if mode == "private":
                images = data.get("images")
                sys_audio = data.get("system_audio")
                if sys_audio:
                    metadata["system_audio"] = sys_audio
            else:
                images = None

        elif msg_type == "text-input":
            user_input = data.get("text", "")
            images = data.get("images")
            metadata = {}
            current_volume = data.get("current_volume")
            if current_volume is not None:
                metadata["current_volume"] = current_volume
            sys_audio = data.get("system_audio")
            if sys_audio and isinstance(sys_audio, dict) and sys_audio.get("data"):
                _b64_len = len(sys_audio["data"])
                logger.info(f"[SysAudio] 프론트→백엔드 수신: {_b64_len}B base64, mime={sys_audio.get('mime_type')}")
                metadata["system_audio"] = sys_audio
            else:
                logger.debug(f"[SysAudio] 프론트에서 미전송 (sys_audio={type(sys_audio).__name__})")

        else:  # mic-audio-end
            user_input = self.received_data_buffers.get(client_uid, np.array([]))
            self.received_data_buffers[client_uid] = np.array([])
            images = data.get("images")
            metadata = {}
            current_volume = data.get("current_volume")
            if current_volume is not None:
                metadata["current_volume"] = current_volume
            sys_audio = data.get("system_audio")
            if sys_audio:
                metadata["system_audio"] = sys_audio

        screen_capture_enabled = (
            context.system_config.enable_screen_capture
            if context and hasattr(context, "system_config")
            else True
        )
        if not screen_capture_enabled:
            images = _filter_images_for_screen_capture_setting(
                images,
                screen_capture_enabled,
                screen_sharing_enabled=getattr(
                    context.system_config, "enable_screen_sharing", False
                ),
            )
            metadata.pop("system_audio", None)

        # 다중 세션: 입력이 들어온 시점의 세션을 꼬리표로 박아둔다.
        # 큐에서 꺼낼 때쯤 사용자가 다른 세션으로 옮겨가 있어도 저장 위치가 안 흔들린다.
        metadata.setdefault(
            "session_id",
            str(data.get("session_id") or self._active_session_id()),
        )

        # letta 는 턴이 끝난 뒤에야 발화를 저장한다. 작업을 시켜 두고 다른 세션에
        # 갔다 돌아왔을 때 방금 시킨 말이 화면에서 사라지지 않도록 세션 보조 로그에
        # 남기는데, 기록 시점은 실제로 턴에 올라가는 순간(_start_turn)이다.
        # 작업 중이라 대기열에 머무는 입력까지 미리 적으면, 아직 전달도 안 된 말이
        # 복원 기록에 먼저 나타나 대기 말풍선과 겹쳐 보인다. (자발 발화는 제외)
        if msg_type == "text-input" and isinstance(user_input, str):
            metadata["_log_user_text"] = True

        await put_immediate({
            "user_input": user_input,
            "images": images,
            "metadata": metadata,
        })

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "character_name": context.character_config.character_name,
                    "human_name": context.character_config.human_name,
                    "client_uid": client_uid,
                }
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")

    async def _handle_tool_approval_response(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """프론트가 보낸 Claude 도구 승인/거부 결정을 중계기로 전달."""
        request_id = data.get("request_id")
        if not request_id:
            logger.warning("tool-approval-response 에 request_id 없음")
            return
        result = {
            "decision": data.get("decision", "deny"),
            "scope": data.get("scope", "once"),
            # '항상 승인'에서 네무린이 고른 범위 옵션 id (hook이 저장 규칙 결정에 사용)
            "rule_option": data.get("rule_option"),
            "reason": data.get("reason", ""),
        }
        self.tool_approval.resolve(request_id, result)

    async def _handle_vtuber_screen_position(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Update the VTuber's current screen position for gaze calculation."""
        try:
            x = float(data.get("x", 0.85))
            y = float(data.get("y", 0.85))
            height = float(data.get("height", 0.5))
            context = self.client_contexts.get(client_uid)
            if context and context.live2d_model:
                context.live2d_model.update_screen_position(x, y, height)
                eye = context.live2d_model._get_eye_screen_position()
                logger.info(
                    f"VTuber position → center=({x:.3f}, {y:.3f}) height={height:.3f} "
                    f"eyeTopRatio={context.live2d_model.model_info.get('eyeTopRatio', 0.2)} "
                    f"eyeLeftRatio={context.live2d_model.model_info.get('eyeLeftRatio', 0.5)} "
                    f"eye=({eye['x']:.3f}, {eye['y']:.3f})"
                )
        except Exception as e:
            logger.error(f"Error updating VTuber screen position: {e}")
