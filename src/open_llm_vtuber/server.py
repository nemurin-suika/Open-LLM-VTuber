"""
Open-LLM-VTuber Server
========================
This module contains the WebSocket server for Open-LLM-VTuber, which handles
the WebSocket connections, serves static files, and manages the web tool.
It uses FastAPI for the server and Starlette for static file serving.
"""

import asyncio
import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

from .routes import init_client_ws_route, init_webtool_routes, init_proxy_route
from .service_context import ServiceContext
from .config_manager.utils import Config


# Create a custom StaticFiles class that adds CORS headers
class CORSStaticFiles(StarletteStaticFiles):
    """
    Static files handler that adds CORS headers to all responses.
    Needed because Starlette StaticFiles might bypass standard middleware.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        # Add CORS headers to all responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"

        return response


class AvatarStaticFiles(CORSStaticFiles):
    """
    Avatar files handler with security restrictions and CORS headers
    """

    async def get_response(self, path: str, scope):
        allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".svg")
        if not any(path.lower().endswith(ext) for ext in allowed_extensions):
            return Response("Forbidden file type", status_code=403)
        response = await super().get_response(path, scope)
        return response


class WebSocketServer:
    """
    API server for Open-LLM-VTuber. This contains the websocket endpoint for the client, hosts the web tool, and serves static files.

    Creates and configures a FastAPI app, registers all routes
    (WebSocket, web tools, proxy) and mounts static assets with CORS.

    Args:
        config (Config): Application configuration containing system settings.
        default_context_cache (ServiceContext, optional):
            Pre‑initialized service context for sessions' service context to reference to.
            **If omitted, `initialize()` method needs to be called to load service context.**

    Notes:
        - If default_context_cache is omitted, call `await initialize()` to load service context cache.
        - Use `clean_cache()` to clear and recreate the local cache directory.
    """

    def __init__(self, config: Config, default_context_cache: ServiceContext = None):
        self.config = config
        self.default_context_cache = default_context_cache or ServiceContext()

        # Create ws_handler now so the lifespan closure can reference it
        client_ws_router, ws_handler = init_client_ws_route(
            default_context_cache=self.default_context_cache
        )

        chzzk_cfg = config.live_config.chzzk_live

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            tasks = []

            # Configure the live chat injector header
            from .live.live_chat_injector import configure as _cfg_injector
            _cfg_injector(header=chzzk_cfg.chat_batch_header)

            # Start the background loop that turns immediate live events into
            # conversation turns (donations, subscriptions, timer-flushed batches)
            tasks.append(asyncio.create_task(ws_handler.start_live_event_loop()))

            # Start Chzzk live platform if a channel is configured
            if chzzk_cfg.channel_id:
                try:
                    from .live.chzzk_live import ChzzkLivePlatform
                    platform = ChzzkLivePlatform(
                        channel_id=chzzk_cfg.channel_id,
                        authorization_key=chzzk_cfg.authorization_key,
                        session_key=chzzk_cfg.session_key,
                        handle_chat=chzzk_cfg.handle_chat,
                        handle_donation=chzzk_cfg.handle_donation,
                        handle_mission=chzzk_cfg.handle_mission,
                        handle_subscription=chzzk_cfg.handle_subscription,
                        chat_timeout=chzzk_cfg.chat_timeout,
                        donation_format=chzzk_cfg.donation_format,
                        mission_format=chzzk_cfg.mission_format,
                        subscription_format=chzzk_cfg.subscription_format,
                    )
                    tasks.append(asyncio.create_task(platform.run()))
                    logger.info(
                        f"Chzzk Live started for channel: {chzzk_cfg.channel_id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to start Chzzk Live: {e}")

            yield  # server runs here

            # Shutdown: cancel all background tasks
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.app = FastAPI(title="Open-LLM-VTuber Server", lifespan=lifespan)

        # Add global CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Include routes
        self.app.include_router(client_ws_router)
        self.app.include_router(
            init_webtool_routes(default_context_cache=self.default_context_cache),
        )

        # Initialize and include proxy routes if proxy is enabled
        system_config = config.system_config
        if hasattr(system_config, "enable_proxy") and system_config.enable_proxy:
            host = system_config.host
            port = system_config.port
            server_url = f"ws://{host}:{port}/client-ws"
            self.app.include_router(
                init_proxy_route(server_url=server_url),
            )

        # Mount cache directory first (to ensure audio file access)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        self.app.mount(
            "/cache",
            CORSStaticFiles(directory="cache"),
            name="cache",
        )

        # Mount static files with CORS-enabled handlers
        self.app.mount(
            "/live2d-models",
            CORSStaticFiles(directory="live2d-models"),
            name="live2d-models",
        )
        self.app.mount(
            "/bg",
            CORSStaticFiles(directory="backgrounds"),
            name="backgrounds",
        )
        self.app.mount(
            "/avatars",
            AvatarStaticFiles(directory="avatars"),
            name="avatars",
        )

        # Mount web tool directory separately from frontend
        self.app.mount(
            "/web-tool",
            CORSStaticFiles(directory="web_tool", html=True),
            name="web_tool",
        )

        # Mount main frontend last (as catch-all)
        # frontend/ is a symlink to Open-LLM-VTuber-Web; built output is at dist/web/
        frontend_dir = "frontend/dist/web"
        if not os.path.exists(frontend_dir):
            frontend_dir = "frontend"
        self.app.mount(
            "/",
            CORSStaticFiles(directory=frontend_dir, html=True),
            name="frontend",
        )

    async def initialize(self):
        """Asynchronously load the service context from config.
        Calling this function is needed if default_context_cache was not provided to the constructor."""
        await self.default_context_cache.load_from_config(self.config)

    @staticmethod
    def clean_cache():
        """Clean the cache directory by removing and recreating it."""
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
