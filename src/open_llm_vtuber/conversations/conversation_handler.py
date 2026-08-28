import asyncio
import json
import random
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState
from prompts import prompt_loader

# Per-client proactive speak state: {"count": int}
# count = 누적 발화 횟수 (0 = 아직 한 번도 말한 적 없음 → 새 topic 강제)
_proactive_state: Dict[str, Dict[str, int]] = {}
_PROACTIVE_CONTINUE_PROB = 0.9


def _load_private_variants(raw: str) -> List[str]:
    """private 프롬프트 파일: 섹션 없이 --- 구분된 자유 변형들."""
    return [v.strip() for v in raw.split("---") if v.strip()]


def _pick_private_variant(variants: List[str]) -> str:
    """private 모드는 단순 랜덤 선택 (broadcast의 viewer/solo/continue 분기 불필요)."""
    return random.choice(variants) if variants else "화면 보고 네무린한테 자연스럽게 한 마디 던져봐."


def _build_proactive_user_input(
    context: "ServiceContext",
    client_uid: str,
    mode: str = "broadcast",
) -> str:
    """proactive 트리거 user_input 생성. mode='broadcast' | 'private'.

    broadcast: 기존 viewer/solo/continue 로직 + 시청자수 라인
    private: 단순 자유서술 변형, 시청자수 없음
    """
    prompt_key = "proactive_speak_private_prompt" if mode == "private" else "proactive_speak_prompt"
    prompt_file = (
        context.system_config.tool_prompts.get(prompt_key) if context else None
    )

    if mode == "private":
        if prompt_file:
            try:
                raw = prompt_loader.load_util(prompt_file)
                chosen = _pick_private_variant(_load_private_variants(raw))
            except Exception as e:
                logger.error(f"Private proactive prompt 로드 실패: {e}")
                chosen = "화면 보고 네무린한테 자연스럽게 한 마디 던져봐."
        else:
            chosen = "화면 보고 네무린한테 자연스럽게 한 마디 던져봐."
        return f"[PROACTIVE_SPEAK_TRIGGER mode=private]\n{chosen}\n[/PROACTIVE_SPEAK_TRIGGER]"

    # broadcast (기본)
    if prompt_file:
        try:
            raw = prompt_loader.load_util(prompt_file)
            viewer_v, solo_v, cont_v = _parse_proactive_prompt(raw)
        except Exception as e:
            logger.error(f"Broadcast proactive prompt 로드 실패: {e}")
            viewer_v, solo_v, cont_v = [], ["뭔가 자연스럽게 혼잣말을 해줘."], []
    else:
        viewer_v, solo_v, cont_v = [], ["뭔가 자연스럽게 혼잣말을 해줘."], []
    chosen = _pick_proactive_variant(client_uid, viewer_v, solo_v, cont_v)
    return f"[PROACTIVE_SPEAK_TRIGGER]\n{chosen}\n[/PROACTIVE_SPEAK_TRIGGER]"


def _parse_proactive_prompt(raw: str) -> Tuple[List[str], List[str], List[str]]:
    """Split file into (viewer_variants, solo_variants, continue_variants).

    File format:
      ===VIEWER===   (선택, 없으면 첫 섹션이 viewer로 간주)
      viewer variant 1 --- viewer variant 2
      ===SOLO===
      solo variant 1 --- solo variant 2
      ===CONTINUE===
      continue variant 1 --- continue variant 2
    """
    cont_section = ""
    if "===CONTINUE===" in raw:
        raw, cont_section = raw.split("===CONTINUE===", 1)

    solo_section = ""
    if "===SOLO===" in raw:
        raw, solo_section = raw.split("===SOLO===", 1)

    if "===VIEWER===" in raw:
        _, viewer_section = raw.split("===VIEWER===", 1)
    else:
        viewer_section = raw

    viewer = [v.strip() for v in viewer_section.split("---") if v.strip()]
    solo = [v.strip() for v in solo_section.split("---") if v.strip()]
    cont = [v.strip() for v in cont_section.split("---") if v.strip()]
    return viewer, solo, cont


def _pick_proactive_variant(
    client_uid: str,
    viewer_variants: List[str],
    solo_variants: List[str],
    cont_variants: List[str],
) -> str:
    """첫 호출은 새 topic, 이후엔 90% 확률로 CONTINUE, 10% 새 topic."""
    state = _proactive_state.setdefault(client_uid, {"count": 0})

    if state["count"] > 0 and cont_variants and random.random() < _PROACTIVE_CONTINUE_PROB:
        chosen = random.choice(cont_variants)
        state["count"] += 1
        logger.debug(f"Proactive speak: continue ({state['count']})")
        return chosen

    new_pool = solo_variants if solo_variants else viewer_variants
    chosen = random.choice(new_pool) if new_pool else "뭔가 자연스럽게 혼잣말을 해줘."
    state["count"] += 1
    logger.debug("Proactive speak: new topic")
    return chosen


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
    initial_metadata: Optional[Dict] = None,
) -> None:
    """Handle triggers that start a conversation"""
    metadata = dict(initial_metadata) if initial_metadata else None

    if msg_type == "ai-speak-signal":
        _has_pending = False
        try:
            from ..live.live_chat_injector import has_pending_chats
            _has_pending = has_pending_chats()
        except Exception:
            pass

        if _has_pending:
            logger.debug("Proactive speak: flushing pending viewer chats instead")
            user_input = ""
            metadata = {"from_live": True}
        else:
            mode = data.get("mode", "broadcast")
            if mode not in ("broadcast", "private"):
                mode = "broadcast"
            user_input = _build_proactive_user_input(context, client_uid, mode=mode)
            # 프론트 사이드바 지시문 (websocket_handler와 동일 규칙)
            _instruction = data.get("proactive_instruction")
            if isinstance(_instruction, str) and _instruction.strip():
                user_input = (
                    f"{user_input}\n"
                    f"[PROACTIVE_USER_INSTRUCTION]\n"
                    f"{_instruction.strip()}\n"
                    f"[/PROACTIVE_USER_INSTRUCTION]"
                )
            metadata = {
                "proactive_speak": True,
                "proactive_mode": mode,
                "skip_memory": True,
                "skip_history": True,
            }

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "full-text",
                        "text": "AI wants to speak something...",
                    }
                )
            )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])

    images = data.get("images")
    session_emoji = np.random.choice(EMOJI_LIST)

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations
        task_key = group.group_id
        if (
            task_key not in current_conversation_tasks
            or current_conversation_tasks[task_key].done()
        ):
            logger.info(f"Starting new group conversation for {task_key}")

            current_conversation_tasks[task_key] = asyncio.create_task(
                process_group_conversation(
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_to_group,
                    group_members=group.members,
                    initiator_client_uid=client_uid,
                    user_input=user_input,
                    images=images,
                    session_emoji=session_emoji,
                    metadata=metadata,
                )
            )
    else:
        # Use client_uid as task key for individual conversations
        current_conversation_tasks[client_uid] = asyncio.create_task(
            process_single_conversation(
                context=context,
                websocket_send=websocket.send_text,
                client_uid=client_uid,
                user_input=user_input,
                images=images,
                session_emoji=session_emoji,
                metadata=metadata,
            )
        )


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
):
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=heard_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
