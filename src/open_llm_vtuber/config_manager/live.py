from pydantic import Field
from typing import Dict, ClassVar, List
from .i18n import I18nMixin, Description


class BiliBiliLiveConfig(I18nMixin):
    """Configuration for BiliBili Live platform."""

    room_ids: List[int] = Field([], alias="room_ids")
    sessdata: str = Field("", alias="sessdata")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "room_ids": Description(
            en="List of BiliBili live room IDs to monitor", zh="要监控的B站直播间ID列表"
        ),
        "sessdata": Description(
            en="SESSDATA cookie value for authenticated requests (optional)",
            zh="用于认证请求的SESSDATA cookie值（可选）",
        ),
    }


class ChzzkLiveConfig(I18nMixin):
    """Configuration for Chzzk (치지직) Live platform."""

    channel_id: str = Field("", alias="channel_id")
    # NID_AUT / NID_SES: Naver login cookies (optional, needed for adult channels)
    authorization_key: str = Field("", alias="authorization_key")
    session_key: str = Field("", alias="session_key")
    handle_chat: bool = Field(True, alias="handle_chat")
    handle_donation: bool = Field(True, alias="handle_donation")
    handle_mission: bool = Field(True, alias="handle_mission")
    handle_subscription: bool = Field(True, alias="handle_subscription")
    chat_format: str = Field("[채팅] {username}: {message}", alias="chat_format")
    donation_format: str = Field(
        "[후원] {username}님이 {amount}원 후원: {message}", alias="donation_format"
    )
    mission_format: str = Field(
        "[미션] {username}님의 미션 ({amount}원): {mission}", alias="mission_format"
    )
    subscription_format: str = Field(
        "[구독] {username}님이 {month}개월째 구독 중! {message}", alias="subscription_format"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "channel_id": Description(
            en="Chzzk channel ID to monitor (find it in the channel URL)",
            zh="要监控的치지직频道ID",
        ),
        "authorization_key": Description(
            en="NID_AUT Naver cookie (optional, needed for adult channels)",
            zh="NID_AUT Naver Cookie（可选）",
        ),
        "session_key": Description(
            en="NID_SES Naver cookie (optional, needed for adult channels)",
            zh="NID_SES Naver Cookie（可选）",
        ),
        "handle_chat": Description(
            en="Whether to forward chat messages to VTuber",
            zh="是否将聊天消息转发给VTuber",
        ),
        "handle_donation": Description(
            en="Whether to forward donation events to VTuber",
            zh="是否将后援（打赏）事件转发给VTuber",
        ),
        "handle_mission": Description(
            en="Whether to forward mission donation events to VTuber",
            zh="是否将任务打赏事件转发给VTuber",
        ),
        "handle_subscription": Description(
            en="Whether to forward subscription events to VTuber",
            zh="是否将订阅事件转发给VTuber",
        ),
        "chat_format": Description(
            en="Format for chat messages. Variables: {username}, {message}",
            zh="聊天消息格式。变量：{username}, {message}",
        ),
        "donation_format": Description(
            en="Format for donation messages. Variables: {username}, {amount}, {message}",
            zh="后援消息格式。变量：{username}, {amount}, {message}",
        ),
        "mission_format": Description(
            en="Format for mission messages. Variables: {username}, {amount}, {mission}, {message}",
            zh="任务消息格式。变量：{username}, {amount}, {mission}, {message}",
        ),
        "subscription_format": Description(
            en="Format for subscription messages. Variables: {username}, {month}",
            zh="订阅消息格式。变量：{username}, {month}",
        ),
    }


class LiveConfig(I18nMixin):
    """Configuration for live streaming platforms integration."""

    bilibili_live: BiliBiliLiveConfig = Field(
        BiliBiliLiveConfig(), alias="bilibili_live"
    )
    chzzk_live: ChzzkLiveConfig = Field(ChzzkLiveConfig(), alias="chzzk_live")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "bilibili_live": Description(
            en="Configuration for BiliBili Live platform", zh="B站直播平台配置"
        ),
        "chzzk_live": Description(
            en="Configuration for Chzzk (치지직) Live platform",
            zh="치지직直播平台配置",
        ),
    }
