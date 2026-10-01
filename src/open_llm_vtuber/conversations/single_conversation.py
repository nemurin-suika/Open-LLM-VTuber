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
        _WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
        _dt_now = _dt.now()
        _now = _dt_now.strftime("%Y-%m-%d") + f" ({_WEEKDAYS[_dt_now.weekday()]})" + _dt_now.strftime(" %H:%M")
        input_text = f"[현재 시각: {_now}]\n{input_text}" if input_text else f"[현재 시각: {_now}]"

        # Inject current TTS volume so the LLM knows the baseline for [volume_up/down]
        _current_volume = metadata.get("current_volume") if metadata else None
        if _current_volume is not None:
            _vol_line = f"[현재 TTS 볼륨: {_current_volume}%]"
            input_text = f"{_vol_line}\n{input_text}" if input_text else _vol_line



        # 시스템 오디오 분석 결과 주입 (프론트가 base64 오디오를 보냈을 때)
        _sys_audio = metadata.get("system_audio") if metadata else None
        if _sys_audio and isinstance(_sys_audio, dict) and _sys_audio.get("data"):
            try:
                from ..system_audio_analyzer import analyze_system_audio, set_analyze_audio_url
                _aa_url = getattr(context.system_config, "analyze_audio_url", None) if context else None
                _aa_fallbacks = (
                    getattr(context.system_config, "analyze_audio_fallback_urls", None)
                    if context
                    else None
                ) or []
                if _aa_url or _aa_fallbacks:
                    set_analyze_audio_url(_aa_url or "", list(_aa_fallbacks))
                logger.info(
                    f"[SysAudio] 분석 시작 ({len(_sys_audio['data'])}B → "
                    f"{_aa_url or 'default'}"
                    + (f", 폴백 {', '.join(_aa_fallbacks)}" if _aa_fallbacks else "")
                    + ")"
                )
                _audio_block = await analyze_system_audio(
                    _sys_audio["data"],
                    _sys_audio.get("mime_type", "audio/webm"),
                )
                if _audio_block:
                    logger.info(f"[SysAudio] 분석 성공, [시스템 오디오] 블록 주입 ({len(_audio_block)}자)")
                    _audio_tag = f"[시스템 오디오]\n{_audio_block}\n[/시스템 오디오]"
                    input_text = f"{_audio_tag}\n{input_text}" if input_text else _audio_tag
                else:
                    logger.warning("[SysAudio] 분석 결과 비어있음 (analyze-audio 서버 응답 실패 가능성)")
            except Exception as e:
                logger.warning(f"[SysAudio] 분석 통합 실패 (무시): {e}")

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
            # ADB 스크린샷 실시간 전송: LettaAgent에 websocket_send 주입 → generator yield 우회하여 즉시 전송
            if hasattr(context.agent_engine, "_websocket_send_fn"):
                context.agent_engine._websocket_send_fn = websocket_send

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

                elif (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "progress_update"
                ):
                    # [알림: ...] 중간 진행 알림 — TTS 없이 채팅창에만 표시
                    logger.debug(f"Sending progress update: {output_item.get('text', '')[:80]}")
                    await websocket_send(json.dumps(output_item))

                elif (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "image_display"
                ):
                    # 이미지 표시 — 프론트 채팅창에 이미지 렌더링
                    logger.debug(f"Sending image_display: label={output_item.get('label', '')[:80]}")
                    await websocket_send(json.dumps(output_item))

                elif (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "agent_status"
                ):
                    # 세션 활동 상태 — 프론트 채팅창 마지막에 스피너를 표시하기 위한 이벤트
                    logger.debug(
                        f"Sending agent_status: status={output_item.get('status')} detail={output_item.get('detail', '')[:80]}"
                    )
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
                        matches = _RESTRICT_PATTERN.findall(output_item.display_text.text)
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
        finally:
            # ADB 스크린샷 콜백 해제 (conversation 종료 후 dangling 참조 방지)
            if hasattr(context.agent_engine, "_websocket_send_fn"):
                context.agent_engine._websocket_send_fn = None
        # --- End processing agent response ---

        if not full_response and not tts_manager.task_list:
            logger.warning("⚠️ LLM 응답 없음: Letta가 아무 내용도 반환하지 않았습니다. Letta 서버 로그를 확인하세요.")

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

        # Claude Code 사용량 웹소켓 전송 (캐시 파일에서 읽기, 실패 시 무시)
        # stale 판단: 캐시가 5분(300초) 이상 오래됐으면 전송 건너뜀
        _USAGE_STALE_SEC = 300
        try:
            import json as _json_usage
            import os as _os_usage
            import time as _time_usage
            _usage_path = "/tmp/claude_usage_cache.json"
            if _os_usage.path.exists(_usage_path):
                _mtime = _os_usage.path.getmtime(_usage_path)
                if _time_usage.time() - _mtime <= _USAGE_STALE_SEC:
                    with open(_usage_path, encoding="utf-8") as _uf:
                        _usage_data = _json_usage.load(_uf)
                    await websocket_send(_json_usage.dumps({
                        "type": "claude_usage",
                        "data": _usage_data,
                    }))
                    logger.debug(f"[Conv] claude_usage 전송: {_usage_data}")
                else:
                    logger.debug(f"[Conv] claude_usage 캐시가 오래됨 (stale, skip): {_time_usage.time() - _mtime:.0f}s 경과")
        except Exception as _ue:
            logger.warning(f"[Conv] claude_usage 전송 실패 (무시): {_ue}")

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
