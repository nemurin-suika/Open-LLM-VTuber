"""
Chzzk (치지직) live platform integration.

Runs inside the VTuber server process — no proxy WebSocket required.
Viewer chats are pushed to live_chat_injector.add_chat() and consumed by
process_single_conversation when the user speaks, types, or proactive fires.
Donations, subscriptions, and timer-triggered chat batches go to the
immediate_queue and trigger standalone conversation turns.
"""

import asyncio
import traceback
from typing import Optional

from loguru import logger

try:
    from chzzkpy import ChatClient as ChzzkChatClient
    from chzzkpy.chat.message import (
        ChatMessage,
        DonationMessage,
        SubscriptionMessage,
        MissionDonationExtra,
        MissionParticipationDonation,
    )

    CHZZKPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"chzzkpy import failed: {e}")
    logger.warning("Chzzk live functionality will not be available.")
    CHZZKPY_AVAILABLE = False

from . import live_chat_injector


class ChzzkLivePlatform:
    """
    Connects to a Chzzk channel and feeds events into the VTuber pipeline.

    - Chat messages → live_chat_injector.add_chat() (buffered, combined with user input)
    - Donations / subscriptions → live_chat_injector.put_immediate() (standalone turn)
    - Timer flush → if chats have been waiting ≥ chat_timeout seconds with no user
      interaction, push the batch to the immediate queue for a standalone turn.
    """

    def __init__(
        self,
        channel_id: str,
        authorization_key: str = "",
        session_key: str = "",
        handle_chat: bool = True,
        handle_donation: bool = True,
        handle_mission: bool = True,
        handle_subscription: bool = True,
        chat_format: str = "[채팅] {username}: {message}",
        donation_format: str = "[후원] {username}님이 {amount}원 후원: {message}",
        mission_format: str = "[미션] {username}님의 미션 ({amount}원): {mission}",
        subscription_format: str = "[구독] {username}님이 {month}개월째 구독 중!",
    ):
        if not CHZZKPY_AVAILABLE:
            raise ImportError(
                "chzzkpy is required for Chzzk live functionality. "
                "Install with: uv add chzzkpy"
            )

        self._channel_id = channel_id
        self._authorization_key = authorization_key or None
        self._session_key = session_key or None
        self._handle_chat = handle_chat
        self._handle_donation = handle_donation
        self._handle_mission = handle_mission
        self._handle_subscription = handle_subscription
        self._chat_format = chat_format
        self._donation_format = donation_format
        self._mission_format = mission_format
        self._subscription_format = subscription_format
        self._running = False
        self._chat_client: Optional[ChzzkChatClient] = None

    # ──────────────────────────────────────────────
    # Event setup
    # ──────────────────────────────────────────────

    def _get_nickname(self, message) -> str:
        try:
            if hasattr(message, "profile") and message.profile:
                return message.profile.nickname or "익명"
        except Exception:
            pass
        return "익명"

    def _setup_events(self, client: "ChzzkChatClient") -> None:
        if self._handle_chat:

            @client.event
            async def on_chat(message: ChatMessage):
                try:
                    username = self._get_nickname(message)
                    content = getattr(message, "content", "") or ""
                    user_id = getattr(message, "user_id", None) or ""
                    logger.info(f"[Chzzk Chat] {username}: {content}")
                    live_chat_injector.register_user(username, user_id)
                    text = self._chat_format.format(username=username, message=content)
                    await live_chat_injector.put_immediate(text)
                except Exception as e:
                    logger.error(f"Error handling chat: {e}")

        if self._handle_donation or self._handle_mission:

            @client.event
            async def on_donation(message: DonationMessage):
                try:
                    username = self._get_nickname(message)
                    extras = getattr(message, "extras", None)
                    content = getattr(message, "content", "") or ""

                    # Determine if mission
                    is_mission = False
                    if self._handle_mission and extras is not None:
                        is_mission = isinstance(
                            extras, (MissionDonationExtra, MissionParticipationDonation)
                        )
                        if not is_mission and hasattr(extras, "donation_type"):
                            is_mission = str(extras.donation_type).upper() == "MISSION"

                    if is_mission:
                        mission_text = getattr(extras, "mission_text", content) or content
                        amount = getattr(extras, "pay_amount", 0) or 0
                        mission_username = getattr(extras, "nickname", None) or username
                        logger.info(
                            f"[Chzzk Mission] {mission_username}: {mission_text} ({amount}원)"
                        )
                        event_text = self._mission_format.format(
                            username=mission_username,
                            mission=mission_text,
                            amount=amount,
                            message=content,
                        )
                    elif self._handle_donation:
                        amount = getattr(extras, "pay_amount", 0) if extras else 0
                        logger.info(f"[Chzzk Donation] {username}: {amount}원")
                        event_text = self._donation_format.format(
                            username=username, amount=amount, message=content
                        )
                    else:
                        return

                    await live_chat_injector.put_immediate(event_text)
                except Exception as e:
                    logger.error(f"Error handling donation: {e}")

        if self._handle_subscription:

            @client.event
            async def on_subscription(message: SubscriptionMessage):
                try:
                    username = self._get_nickname(message)
                    extras = getattr(message, "extras", None)
                    month = getattr(extras, "month", 1) if extras else 1
                    logger.info(f"[Chzzk Subscription] {username}: {month}개월")
                    event_text = self._subscription_format.format(
                        username=username, month=month
                    )
                    await live_chat_injector.put_immediate(event_text)
                except Exception as e:
                    logger.error(f"Error handling subscription: {e}")

        @client.event
        async def on_connect():
            logger.info(f"Connected to Chzzk Live channel: {self._channel_id}")

        @client.event
        async def on_client_error(error, *args):
            logger.error(f"Chzzk Live error: {error}")

    async def _viewer_count_poll_loop(self) -> None:
        from chzzkpy import Client as ChzzkHttpClient
        logger.info("Viewer count polling started (interval: 60s)")
        while self._running:
            try:
                client_kwargs = {}
                if self._authorization_key:
                    client_kwargs["authorization_key"] = self._authorization_key
                if self._session_key:
                    client_kwargs["session_key"] = self._session_key
                async with ChzzkHttpClient(**client_kwargs) as client:
                    status = await client.live_status(self._channel_id)
                    count = status.concurrent_user_count if status else 0
                    live_chat_injector.set_viewer_count(count)
                    logger.debug(f"[Chzzk] Viewer count updated: {count}")
            except Exception as e:
                logger.warning(f"[Chzzk] Viewer count poll failed: {e}")
            await asyncio.sleep(60)

    # ──────────────────────────────────────────────
    # Moderation
    # ──────────────────────────────────────────────

    async def _do_restrict(self, nickname: str) -> bool:
        """Temporarily restrict a viewer by nickname. Requires login credentials."""
        user_id = live_chat_injector.get_user_id(nickname)
        if not user_id:
            logger.warning(f"[Chzzk Restrict] Unknown user: '{nickname}' (not seen in chat yet)")
            return False
        if not self._chat_client:
            logger.warning("[Chzzk Restrict] Chat client not ready")
            return False
        try:
            await self._chat_client.temporary_restrict(user_id)
            logger.info(f"[Chzzk Restrict] Temporarily restricted: {nickname}")
            return True
        except Exception as e:
            logger.error(f"[Chzzk Restrict] Failed to restrict '{nickname}': {e}")
            return False

    # ──────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        live_chat_injector.set_restrict_callback(self._do_restrict)
        logger.info(f"Connecting to Chzzk Live channel: {self._channel_id}")

        retry_delay = 5

        try:
            while self._running:
                client = ChzzkChatClient(channel_id=self._channel_id)
                self._chat_client = client
                self._setup_events(client)
                viewer_poll_task = asyncio.create_task(self._viewer_count_poll_loop())
                try:
                    await client.start(
                        authorization_key=self._authorization_key,
                        session_key=self._session_key,
                    )
                    break
                except asyncio.CancelledError:
                    logger.info("Chzzk Live platform cancelled")
                    return
                except Exception as e:
                    logger.error(f"Chzzk Live error: {e}")
                    logger.debug(traceback.format_exc())
                    if self._running:
                        logger.info(f"Reconnecting in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                finally:
                    viewer_poll_task.cancel()
                    try:
                        await viewer_poll_task
                    except asyncio.CancelledError:
                        pass
        finally:
            self._running = False
            self._chat_client = None
            logger.info("Chzzk Live platform stopped")
