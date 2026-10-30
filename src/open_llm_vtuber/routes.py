import os
import json
from uuid import uuid4
import numpy as np
from datetime import datetime
from fastapi import APIRouter, WebSocket, UploadFile, File, Response, Request
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler
from .user_identity import extract_identity


def init_client_ws_route(
    default_context_cache: ServiceContext,
) -> tuple:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        tuple: (APIRouter, WebSocketHandler) — the handler is exposed so the
               server can wire up live-platform background tasks.
    """

    router = APIRouter()
    ws_handler = WebSocketHandler(default_context_cache)

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        # 신원 헤더는 핸드셰이크에만 실려 오므로 accept 전에 읽어둔다(TODO2 1-10).
        identity = extract_identity(websocket.headers)
        await websocket.accept()
        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(
                websocket, client_uid, identity=identity
            )
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router, ws_handler


def init_tool_approval_route(ws_handler: WebSocketHandler) -> APIRouter:
    """Claude 도구 사용 승인 중계 HTTP 엔드포인트.

    letta가 띄운 `claude -p` 의 PreToolUse hook이 여기로 POST하면,
    연결된 프론트로 승인 요청을 중계하고 네무린의 결정을 기다린 뒤 돌려준다.
    """

    router = APIRouter()

    @router.post("/tool-approval")
    async def tool_approval(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"decision": "deny", "reason": "bad_request"}, status_code=400)

        # 다중 세션: hook 이 실어 보낸 세션 키를 conversation_id + 사용자가 정한
        # 이름으로 환산해, 팝업이 "어느 방의 요청인지" 보여줄 수 있게 한다.
        if isinstance(payload, dict):
            try:
                session_id, session_name = await ws_handler.resolve_session_label(
                    str(payload.get("session_key") or "")
                )
                payload["session_id"] = session_id
                payload["session_name"] = session_name
            except Exception as e:
                logger.debug(f"[ToolApproval] 세션 라벨 결정 실패: {e}")

        # 승인 팝업은 주인(네무린)에게만 띄운다. 손님 턴은 hook 단계에서 이미
        # 전부 거절되므로 여기까지 오지 않고, 와도 손님 화면에는 안 뜬다.
        result = await ws_handler.tool_approval.request_approval(
            clients=ws_handler.owner_connections,
            payload=payload,
        )
        return JSONResponse(result)

    return router


def init_session_status_route(ws_handler: WebSocketHandler) -> APIRouter:
    """세션 작업 상태 조회 엔드포인트 (TODO26).

    미즈키가 letta/백엔드를 재기동하기 전에 `curl -s -m 5
    http://127.0.0.1:12393/session-status` 로 먼저 물어보고, 다른 세션이
    턴을 돌고 있으면 재기동을 멈추기 위한 창구다.

    재기동은 돌고 있던 턴을 통째로 끊어버리기 때문에, 파일을 반만 고친
    상태로 남을 수 있다. 백엔드가 이미 죽어 있으면 이 요청도 실패하는데,
    그건 돌고 있는 턴도 없다는 뜻이라 그냥 재기동해도 된다.
    """

    router = APIRouter()

    @router.get("/session-status")
    async def session_status():
        try:
            return JSONResponse(ws_handler.session_status())
        except Exception as e:
            logger.error(f"[session-status] 상태 조회 실패: {e}")
            # 판단 근거가 없을 땐 "안전하지 않다"로 답해서 재기동을 멈추게 한다.
            return JSONResponse(
                {"error": str(e), "safe_to_restart": False}, status_code=500
            )

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        server_url: The WebSocket URL of the actual server

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/live2d-models/info")
    async def get_live2d_folder_info():
        """Get information about available Live2D models"""
        live2d_dir = "live2d-models"
        if not os.path.exists(live2d_dir):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = []
        supported_extensions = [".png", ".jpg", ".jpeg"]

        for entry in os.scandir(live2d_dir):
            if entry.is_dir():
                folder_name = entry.name.replace("\\", "/")
                model3_file = os.path.join(
                    live2d_dir, folder_name, f"{folder_name}.model3.json"
                ).replace("\\", "/")

                if os.path.isfile(model3_file):
                    # Find avatar file if it exists
                    avatar_file = None
                    for ext in supported_extensions:
                        avatar_path = os.path.join(
                            live2d_dir, folder_name, f"{folder_name}{ext}"
                        )
                        if os.path.isfile(avatar_path):
                            avatar_file = avatar_path.replace("\\", "/")
                            break

                    valid_characters.append(
                        {
                            "name": folder_name,
                            "avatar": avatar_file,
                            "model_path": model3_file,
                        }
                    )
        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/asr")
    async def transcribe_audio(file: UploadFile = File(...)):
        """
        Endpoint for transcribing audio using the ASR engine
        """
        logger.info(f"Received audio file for transcription: {file.filename}")

        if default_context_cache.asr_engine is None:
            logger.warning("Transcription requested but ASR is disabled.")
            return Response(
                content=json.dumps(
                    {"error": "ASR is disabled (asr_model is not configured)"}
                ),
                status_code=503,
                media_type="application/json",
            )

        try:
            contents = await file.read()

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Decode the WAV header and get actual audio data
            wav_header_size = 44  # Standard WAV header size
            audio_data = contents[wav_header_size:]

            # Validate audio data size
            if len(audio_data) % 2 != 0:
                raise ValueError("Invalid audio data: Buffer size must be even")

            # Convert to 16-bit PCM samples to float32
            try:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except ValueError as e:
                raise ValueError(
                    f"Audio format error: {str(e)}. Please ensure the file is 16-bit PCM WAV format."
                )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            logger.info(f"Transcription result: {text}")
            return {"text": text}

        except ValueError as e:
            logger.error(f"Audio format error: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """WebSocket endpoint for TTS generation"""
        await websocket.accept()
        logger.info("TTS WebSocket connection established")

        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text")
                if not text:
                    continue

                logger.info(f"Received text for TTS: {text}")

                # Split text into sentences
                sentences = [s.strip() for s in text.split(".") if s.strip()]

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        sentence = sentence + "."  # Add back the period
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        logger.info(
                            f"Generated audio for sentence: {sentence} at: {audio_path}"
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.info("TTS WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket connection: {e}")
            await websocket.close()

    return router
