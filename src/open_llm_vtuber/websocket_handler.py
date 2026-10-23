from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import os
import time
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

        # 현재 실행 중인 LLM+TTS capture task 참조 (인터럽트 시 cancel 대상)
        self._current_capture_task: Optional[asyncio.Task] = None

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
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

        # Ordered playback queues: each slot is (messages: list, ready: Event)
        regular_pq: asyncio.Queue = asyncio.Queue()
        priority_pq: asyncio.Queue = asyncio.Queue()

        # ── helpers ──────────────────────────────────────────────────────────

        async def _capture(item: dict, messages: list, ready: asyncio.Event) -> None:
            """Run LLM+TTS in capture mode; fill messages, then set ready."""
            # 현재 task를 추적 — 인터럽트 시 cancel 대상
            self._current_capture_task = asyncio.current_task()
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
                base_meta["skip_playback_wait"] = True

                captured: list[str] = []
                _seen_notify_ts: set = set()

                # [알림: ...] 중간 진행 알림을 캡처 버퍼 우회해 실시간 전송
                import os as _os_notify
                # 다중 세션: 알림 파일은 letta conversation 단위로 분리되어 있다.
                # 이 클라이언트가 쓰는 agent의 경로를 그대로 따라간다(없으면 단일 세션 기본 경로).
                _NOTIFY_FILE = getattr(
                    getattr(context, "agent_engine", None),
                    "_notify_file",
                    "/tmp/mizuki_progress_notify.jsonl",
                )

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
                                                await ws.send_text(json.dumps({
                                                    "type": "progress_update",
                                                    "text": _entry.get("text", ""),
                                                    "timestamp": _ts,
                                                }))
                                        except Exception:
                                            pass
                        except Exception:
                            pass
                        await asyncio.sleep(0.3)

                poll_task = asyncio.create_task(_poll_and_send_notify())

                async def _cap_send(raw: str) -> None:
                    # progress_update는 poll_task가 실시간 전송 → captured 제외
                    # image_display는 오디오 순서와 무관하므로 즉시 전송 → captured 제외
                    try:
                        _msg_type = json.loads(raw).get("type")
                        if _msg_type == "progress_update":
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
                if self._current_capture_task is asyncio.current_task():
                    self._current_capture_task = None
                ready.set()

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

        async def _replay(messages: list, idx: int = 0) -> None:
            """Send captured messages with progressive audio delivery.

            In normal (non-capture) mode audio chunks arrive one at a time as TTS
            generates (~2-3s per sentence). In replay mode all chunks would arrive
            within milliseconds, causing the frontend to play them simultaneously.
            Progressive delivery adds a per-chunk delay equal to its duration so the
            frontend receives chunks at the same pace as normal mode.
            """
            if not messages or not self.client_connections:
                return
            client_uid = next(reversed(self.client_connections))
            ws = self.client_connections.get(client_uid)
            if not ws:
                return

            total_audio_ms = 0
            audio_chunk_count = 0
            send_failed = False
            try:
                for raw in messages:
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
                            await asyncio.sleep(chunk_ms / 1000)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[TTS-DBG #{idx}] Error during replay send: {e}")
                send_failed = True
            finally:
                if total_audio_ms > 0:
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

        async def _next_slot(prio: asyncio.Queue, regular: asyncio.Queue):
            """Return next (messages, ready) slot, always preferring prio."""
            # Non-blocking check first
            try:
                return prio.get_nowait()
            except asyncio.QueueEmpty:
                pass

            prio_task = asyncio.create_task(prio.get())
            reg_task = asyncio.create_task(regular.get())
            done, pending = await asyncio.wait(
                {prio_task, reg_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in list(pending):
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            if prio_task in done:
                if reg_task in done:
                    regular.put_nowait(reg_task.result())
                return prio_task.result()

            # Regular won — check priority one more time before accepting it
            try:
                slot = prio.get_nowait()
                regular.put_nowait(reg_task.result())
                return slot
            except asyncio.QueueEmpty:
                return reg_task.result()

        # ── player ───────────────────────────────────────────────────────────

        _play_idx = 0  # monotonic counter to correlate log lines per output

        async def _player() -> None:
            """먼저 완성된 슬롯을 먼저 재생한다 (completion-order playback)."""
            import time
            nonlocal _play_idx
            SLOT_TIMEOUT = 7200.0  # 슬롯당 최대 대기 시간 (초) — 2시간
            # pending: (messages, ready_event, enqueued_at)
            pending: list[tuple[list, asyncio.Event, float]] = []

            def _drain_queues() -> None:
                """큐에서 즉시 꺼낼 수 있는 슬롯을 모두 pending에 추가."""
                while True:
                    try:
                        msgs, rdy = priority_pq.get_nowait()
                        pending.insert(0, (msgs, rdy, time.monotonic()))
                    except asyncio.QueueEmpty:
                        break
                while True:
                    try:
                        msgs, rdy = regular_pq.get_nowait()
                        pending.append((msgs, rdy, time.monotonic()))
                    except asyncio.QueueEmpty:
                        break

            def _purge_timed_out() -> None:
                """300초가 지난 슬롯을 버린다."""
                now = time.monotonic()
                for i in reversed(range(len(pending))):
                    _, _, enqueued_at = pending[i]
                    if now - enqueued_at > SLOT_TIMEOUT:
                        pending.pop(i)
                        logger.warning("[TTS-DBG] ⏰ slot timed out (LLM/TTS >7200s) — skipping")
                        # slot timeout 시 claude -p 프로세스도 종료
                        try:
                            if self.client_connections:
                                _uid = next(reversed(self.client_connections))
                                _ctx = self.client_contexts.get(_uid)
                                if _ctx and hasattr(_ctx, "agent_engine"):
                                    _agent = _ctx.agent_engine
                                    if hasattr(_agent, "handle_interrupt"):
                                        _agent.handle_interrupt("")
                                        logger.info("[TTS-DBG] slot timeout → handle_interrupt() 호출")
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
                        msgs, rdy = get_prio.result()
                        pending.insert(0, (msgs, rdy, now))
                        if get_reg in done:
                            msgs, rdy = get_reg.result()
                            pending.append((msgs, rdy, now))
                    else:
                        msgs, rdy = get_reg.result()
                        pending.append((msgs, rdy, now))
                    continue

                # ── 이미 완성된 슬롯이 있는지 즉시 확인 ─────────────────────────
                ready_idx = next(
                    (i for i, (_, rdy, _t) in enumerate(pending) if rdy.is_set()), None
                )

                if ready_idx is None:
                    # 타임아웃까지 남은 시간 계산 (가장 오래된 슬롯 기준)
                    now = time.monotonic()
                    wait_timeout = max(
                        0.1,
                        SLOT_TIMEOUT - max(now - t for _, _, t in pending)
                    )

                    ready_waits = [asyncio.create_task(rdy.wait()) for _, rdy, _ in pending]
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
                            msgs, rdy = get_prio.result()
                            pending.insert(0, (msgs, rdy, now))
                        except Exception:
                            pass
                    if get_reg in done:
                        try:
                            msgs, rdy = get_reg.result()
                            pending.append((msgs, rdy, now))
                        except Exception:
                            pass

                    ready_idx = next(
                        (i for i, t in enumerate(ready_waits) if t in done), None
                    )
                    if ready_idx is None:
                        continue  # 새 슬롯 도착 또는 타임아웃 체크, 다시 루프

                # ── 완성된 슬롯 재생 ──────────────────────────────────────────────
                msgs, _, _ = pending.pop(ready_idx)
                _play_idx += 1
                idx = _play_idx
                logger.info(
                    f"[TTS-DBG #{idx}] ▶ replay start  "
                    f"(pq_regular={regular_pq.qsize()} pq_priority={priority_pq.qsize()} pending={len(pending)})"
                )
                try:
                    await _replay(msgs, idx)
                    logger.info(f"[TTS-DBG #{idx}] ✓ replay done")
                    mark_turn_stage("delivered")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Live playback error: {e}")

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
                await priority_pq.put((messages, ready))
                asyncio.create_task(_capture(item, messages, ready))

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
                await regular_pq.put((msgs, rdy))
                asyncio.create_task(_capture(batch_item, msgs, rdy))

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
                    await regular_pq.put((msgs, rdy))
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

                msgs: list = []
                rdy = asyncio.Event()
                await regular_pq.put((msgs, rdy))
                asyncio.create_task(_capture(item, msgs, rdy))

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

        # 1. 현재 capture task 취소
        task = self._current_capture_task
        if task and not task.done():
            logger.info("[Interrupt] 현재 capture task 취소")
            task.cancel()

        # 2. letta agent abort 플래그 설정
        context = self.client_contexts.get(client_uid)
        if context and hasattr(context, "agent_engine"):
            agent = context.agent_engine
            if hasattr(agent, "handle_interrupt"):
                agent.handle_interrupt("")
                logger.info("[Interrupt] agent.handle_interrupt() 호출 완료")

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

        # Drop screenshot images when enable_screen_capture is False
        screen_capture_enabled = (
            context.system_config.enable_screen_capture
            if context and hasattr(context, "system_config")
            else True
        )
        if not screen_capture_enabled:
            images = None
            # 시스템 오디오도 같은 토글로 묶어 비활성화 가능 (별도 토글 추가 시 분리)
            metadata.pop("system_audio", None)

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
