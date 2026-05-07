from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
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

        _OVERLOAD_PATTERNS = [
            "[오버로드] 채팅이 갑자기 너무 많이 쏟아져서 다 읽기가 힘들어. 시청자들한테 너무 많이 온다고 자연스럽게 반응해줘.",
            "[오버로드] 채팅 화력이 감당이 안 될 정도로 쏟아지고 있어. 당황하거나 압도당한 느낌으로 짧게 반응해줘.",
            "[오버로드] 채팅이 한꺼번에 너무 많이 들어와서 멈춰야 했어. 읽다가 뒤처진 상황에 맞게 반응해줘.",
            "[오버로드] 갑자기 채팅 폭주가 왔어. 시청자들 화력 칭찬하면서 좀 진정시켜달라고 해줘.",
            "[오버로드] 채팅이 너무 빠르게 올라와서 하나도 못 읽겠어. 이 상황 솔직하게 토로해줘.",
        ]

        # Ordered playback queues: each slot is (messages: list, ready: Event)
        regular_pq: asyncio.Queue = asyncio.Queue()
        priority_pq: asyncio.Queue = asyncio.Queue()

        # ── helpers ──────────────────────────────────────────────────────────

        async def _capture(item: dict, messages: list, ready: asyncio.Event) -> None:
            """Run LLM+TTS in capture mode; fill messages, then set ready."""
            user_input_preview = str(item.get("user_input", ""))[:40]
            logger.debug(f"[TTS-DBG] capture start: {user_input_preview!r}")
            try:
                if not self.client_connections:
                    return
                client_uid = next(reversed(self.client_connections))
                context = self.client_contexts.get(client_uid)
                if not context:
                    return

                user_input = item.get("user_input", "")
                images = item.get("images")
                base_meta = dict(item.get("metadata") or {})
                base_meta["skip_playback_wait"] = True

                captured: list[str] = []

                async def _cap_send(raw: str) -> None:
                    captured.append(raw)

                await process_single_conversation(
                    context=context,
                    websocket_send=_cap_send,
                    client_uid=client_uid,
                    user_input=user_input,
                    images=images,
                    metadata=base_meta,
                )
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
                ready.set()

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
            nonlocal _play_idx
            while True:
                messages, ready = await _next_slot(priority_pq, regular_pq)
                try:
                    await asyncio.wait_for(ready.wait(), timeout=300.0)
                except asyncio.TimeoutError:
                    logger.warning("[TTS-DBG] ⏰ slot timed out (LLM/TTS >300s) — skipping")
                    continue
                _play_idx += 1
                idx = _play_idx
                logger.info(f"[TTS-DBG #{idx}] ▶ replay start  (pq_regular={regular_pq.qsize()} pq_priority={priority_pq.qsize()})")
                try:
                    await _replay(messages, idx)
                    logger.info(f"[TTS-DBG #{idx}] ✓ replay done")
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
            BATCH_HEADER = "[시청자 채팅]"

            chat_buffer: list[str] = []  # Accumulates live chat texts during buffer mode

            async def _flush_chat_buffer() -> None:
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
                    await _flush_chat_buffer()  # flush before overload message
                    overload_text = random.choice(_OVERLOAD_PATTERNS)
                    item = {"user_input": overload_text, "images": None,
                            "metadata": {"from_live": True}}
                    logger.warning(
                        f"Regular queue overloaded ({pending} pending, "
                        f"drained {drained}) — prompting overload response"
                    )

                # Determine if this is a live chat eligible for buffering
                is_live_chat = (
                    isinstance(item.get("user_input"), str)
                    and bool(item.get("metadata", {}).get("from_live"))
                )

                if is_live_chat and regular_pq.qsize() > CHAT_BUFFER_THRESHOLD:
                    # Playback queue is deep — buffer this chat instead of
                    # immediately starting a new LLM call
                    chat_buffer.append(item["user_input"])
                    logger.debug(
                        f"[Chat Buffer] pq={regular_pq.qsize()} "
                        f"→ buffered ({len(chat_buffer)}개): {item['user_input'][:60]}"
                    )
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

        Only the explicit frontend interrupt button should reach here.
        Automatic sources (VAD, text submission) are disabled upstream.
        The frontend already stops its own audio when the button is pressed,
        which sends frontend-playback-complete and naturally unblocks the player.
        """
        logger.debug(f"Interrupt signal received from {client_uid} — no server action taken")

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
            try:
                from prompts import prompt_loader
                from .conversations.conversation_handler import (
                    _parse_proactive_prompt,
                    _pick_proactive_variant,
                )
                prompt_file = context.system_config.tool_prompts.get(
                    "proactive_speak_prompt"
                ) if context else None
                if prompt_file:
                    raw = prompt_loader.load_util(prompt_file)
                    viewer_variants, solo_variants, cont_variants = _parse_proactive_prompt(raw)
                    chosen = _pick_proactive_variant(client_uid, viewer_variants, solo_variants, cont_variants)
                else:
                    chosen = "뭔가 자연스럽게 혼잣말을 해줘."
                user_input = f"[PROACTIVE_SPEAK_TRIGGER]\n{chosen}\n[/PROACTIVE_SPEAK_TRIGGER]"
            except Exception:
                user_input = "[PROACTIVE_SPEAK_TRIGGER]\n뭔가 자연스럽게 혼잣말을 해줘.\n[/PROACTIVE_SPEAK_TRIGGER]"
            metadata = {
                "proactive_speak": True,
                "skip_memory": True,
                "skip_history": True,
            }
            images = None

        elif msg_type == "text-input":
            user_input = data.get("text", "")
            images = data.get("images")
            metadata = {}
            current_volume = data.get("current_volume")
            if current_volume is not None:
                metadata["current_volume"] = current_volume

        else:  # mic-audio-end
            user_input = self.received_data_buffers.get(client_uid, np.array([]))
            self.received_data_buffers[client_uid] = np.array([])
            images = data.get("images")
            metadata = {}
            current_volume = data.get("current_volume")
            if current_volume is not None:
                metadata["current_volume"] = current_volume

        # Drop screenshot images when enable_screen_capture is False
        screen_capture_enabled = (
            context.system_config.enable_screen_capture
            if context and hasattr(context, "system_config")
            else True
        )
        if not screen_capture_enabled:
            images = None

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
