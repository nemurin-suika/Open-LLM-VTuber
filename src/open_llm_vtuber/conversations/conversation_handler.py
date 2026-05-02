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

# Per-client proactive speak state: {"count": int, "max": int}
# count = how many times the current topic has been spoken (0 = start fresh)
# max   = how many times to continue before switching to a new topic
_proactive_state: Dict[str, Dict[str, int]] = {}


def _parse_proactive_prompt(raw: str) -> Tuple[List[str], List[str], List[str]]:
    """Split file into (viewer_variants, solo_variants, continue_variants).

    File format:
      viewer variant 1 --- viewer variant 2
      ===SOLO===
      solo variant 1 --- solo variant 2
      ===CONTINUE===
      continue variant 1 --- continue variant 2
    """
    cont_section = ""
    if "===CONTINUE===" in raw:
        main_part, cont_section = raw.split("===CONTINUE===", 1)
    else:
        main_part = raw

    if "===SOLO===" in main_part:
        viewer_part, solo_part = main_part.split("===SOLO===", 1)
    else:
        viewer_part, solo_part = "", main_part

    viewer = [v.strip() for v in viewer_part.split("---") if v.strip()]
    solo = [v.strip() for v in solo_part.split("---") if v.strip()]
    cont = [v.strip() for v in cont_section.split("---") if v.strip()]
    return viewer, solo, cont


def _pick_proactive_variant(
    client_uid: str,
    viewer_variants: List[str],
    solo_variants: List[str],
    cont_variants: List[str],
) -> str:
    """Pick new-topic or continue variant based on per-client state and viewer count."""
    state = _proactive_state.setdefault(client_uid, {"count": 0, "max": 0})

    if state["count"] > 0 and state["count"] < state["max"] and cont_variants:
        chosen = random.choice(cont_variants)
        state["count"] += 1
        logger.debug(f"Proactive speak: continuing topic ({state['count']}/{state['max']})")
    else:
        try:
            from ..live.live_chat_injector import get_viewer_count
            vc = get_viewer_count()
        except Exception:
            vc = -1

        # Only use viewer-targeting prompts when 10+ viewers are watching
        if vc >= 10 and viewer_variants:
            new_pool = viewer_variants + solo_variants
        else:
            new_pool = solo_variants if solo_variants else viewer_variants

        chosen = random.choice(new_pool) if new_pool else "뭔가 자연스럽게 혼잣말을 해줘."
        state["max"] = random.randint(1, 3)
        state["count"] = 1
        logger.debug(f"Proactive speak: new topic, viewer_count={vc} (will continue up to {state['max']} times)")

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
            try:
                prompt_name = "proactive_speak_prompt"
                prompt_file = context.system_config.tool_prompts.get(prompt_name)
                if prompt_file:
                    raw = prompt_loader.load_util(prompt_file)
                    viewer_variants, solo_variants, cont_variants = _parse_proactive_prompt(raw)
                else:
                    logger.warning("Proactive speak prompt not configured, using default")
                    viewer_variants, solo_variants, cont_variants = [], ["뭔가 자연스럽게 혼잣말을 해줘."], []
                chosen = _pick_proactive_variant(client_uid, viewer_variants, solo_variants, cont_variants)

                try:
                    from ..live.live_chat_injector import get_viewer_count
                    vc = get_viewer_count()
                    vc_line = f"[실시간 시청자: {vc}명]\n" if vc >= 0 else ""
                except Exception:
                    vc_line = ""

                user_input = f"{vc_line}[PROACTIVE_SPEAK_TRIGGER]\n{chosen}\n[/PROACTIVE_SPEAK_TRIGGER]"
            except Exception as e:
                logger.error(f"Error loading proactive speak prompt: {e}")
                user_input = "[PROACTIVE_SPEAK_TRIGGER]\n뭔가 자연스럽게 혼잣말을 해줘.\n[/PROACTIVE_SPEAK_TRIGGER]"

            metadata = {
                "proactive_speak": True,
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
