from typing import Union, List, Dict, Any, Optional
import asyncio
import json
import re
from loguru import logger
import numpy as np

# Pattern the LLM uses to request a temporary chat restriction.
# Example: [채팅제한 username]
_RESTRICT_PATTERN = re.compile(r"\[채팅제한\s+([^\]]+?)\]")

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()
    full_response = ""  # Initialize full_response here

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )

        # Wrap nemurin's direct input so it's distinguishable from viewer chats
        is_proactive = metadata and metadata.get("proactive_speak", False)
        is_from_live = metadata and metadata.get("from_live", False)
        if input_text and not is_proactive and not is_from_live:
            _human = context.character_config.human_name
            input_text = f"[{_human}]\n{input_text}\n[/{_human}]"

        # Inject current date/time so the LLM always knows when the conversation is
        from datetime import datetime as _dt
        _now = _dt.now().strftime("%Y-%m-%d %H:%M")
        input_text = f"[현재 시각: {_now}]\n{input_text}" if input_text else f"[현재 시각: {_now}]"

        # Inject current TTS volume so the LLM knows the baseline for [volume_up/down]
        _current_volume = metadata.get("current_volume") if metadata else None
        if _current_volume is not None:
            _vol_line = f"[현재 TTS 볼륨: {_current_volume}%]"
            input_text = f"{_vol_line}\n{input_text}" if input_text else _vol_line

        # Inject real-time viewer count when the stream is live (count > 0)
        try:
            from ..live.live_chat_injector import get_viewer_count as _get_viewer_count
            _vc = _get_viewer_count()
            if _vc > 0:
                _vc_line = f"[실시간 시청자: {_vc}명]"
                input_text = f"{_vc_line}\n{input_text}" if input_text else _vc_line
        except Exception:
            pass

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
            metadata=metadata,
        )

        skip_history = metadata and metadata.get("skip_history", False)
        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        # In capture mode (parallel LLM), skip waiting for frontend-playback-complete.
        # The captured messages are replayed later by the ordered player.
        skip_playback_wait = bool(metadata and metadata.get("skip_playback_wait", False))

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        try:
            # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
            agent_output_stream = context.agent_engine.chat(batch_input)

            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):

                    # [IGNORE] 감지: TTS 없이 즉시 종료
                    if isinstance(output_item, SentenceOutput) and not tts_manager.task_list:
                        if output_item.tts_text.strip() == "[IGNORE]":
                            logger.debug("LLM responded with [IGNORE], skipping turn")
                            await websocket_send(json.dumps({"type": "control", "text": "conversation-chain-end"}))
                            return ""

                    # [채팅제한 username] 패턴 감지 및 처리
                    if isinstance(output_item, SentenceOutput):
                        matches = _RESTRICT_PATTERN.findall(output_item.tts_text)
                        if matches:
                            for username in matches:
                                username = username.strip()
                                logger.info(f"LLM requested chat restriction: '{username}'")
                                try:
                                    from ..live.live_chat_injector import execute_restrict
                                    asyncio.create_task(execute_restrict(username))
                                except Exception:
                                    pass
                            # Strip the tag from TTS (don't speak it)
                            output_item.tts_text = _RESTRICT_PATTERN.sub(
                                "", output_item.tts_text
                            ).strip()

                    # Handle SentenceOutput or AudioOutput
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        # Wait for any pending TTS tasks.
        # First gather() waits for all TTS audio files to be generated.
        # Then wait_for_delivery() waits for _sender_task to finish sending ALL
        # payloads to websocket_send — without this, some audio messages would
        # still be in _payload_queue when backend-synth-complete is appended,
        # causing TTS overlap in the capture pipeline.
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await tts_manager.wait_for_delivery()
            await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
            skip_playback_wait=skip_playback_wait,
        )

        if not skip_history and context.history_uid:
            if input_text:
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="human",
                    content=input_text,
                    name=context.character_config.human_name,
                )
            if full_response:
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="ai",
                    content=full_response,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )
                logger.info(f"AI response: {full_response}")

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
