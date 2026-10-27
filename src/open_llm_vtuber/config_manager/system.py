# config_manager/system.py
from pydantic import Field, model_validator, BaseModel
from typing import Dict, List, ClassVar
from .i18n import I18nMixin, Description


class WindEffectConfig(BaseModel):
    """Wind effect configuration for Live2D physics (hair/cloth flutter)."""

    enabled: bool = Field(True, alias="enabled")
    max_strength: float = Field(0.5, alias="max_strength")
    transition_speed: float = Field(0.5, alias="transition_speed")
    interval_min: float = Field(3.0, alias="interval_min")
    interval_max: float = Field(8.0, alias="interval_max")

    model_config = {"populate_by_name": True}


class IdleGazeConfig(BaseModel):
    """Idle gaze wandering configuration for Live2D (active when AI is not responding)."""

    enabled: bool = Field(True, alias="enabled")
    range_x: float = Field(0.3, alias="range_x")
    range_y: float = Field(0.2, alias="range_y")
    transition_speed: float = Field(0.8, alias="transition_speed")
    interval_min: float = Field(2.0, alias="interval_min")
    interval_max: float = Field(5.0, alias="interval_max")

    model_config = {"populate_by_name": True}


class SystemConfig(I18nMixin):
    """System configuration settings."""

    conf_version: str = Field(..., alias="conf_version")
    host: str = Field(..., alias="host")
    port: int = Field(..., alias="port")
    config_alts_dir: str = Field(..., alias="config_alts_dir")
    tool_prompts: Dict[str, str] = Field(..., alias="tool_prompts")
    enable_proxy: bool = Field(False, alias="enable_proxy")
    enable_screen_capture: bool = Field(True, alias="enable_screen_capture")
    enable_screen_sharing: bool = Field(False, alias="enable_screen_sharing")
    analyze_audio_url: str = Field(
        "http://192.168.0.15:9881", alias="analyze_audio_url"
    )
    analyze_audio_fallback_urls: List[str] = Field(
        default_factory=list, alias="analyze_audio_fallback_urls"
    )
    wind_effect: WindEffectConfig = Field(
        default_factory=WindEffectConfig, alias="wind_effect"
    )
    idle_gaze: IdleGazeConfig = Field(
        default_factory=IdleGazeConfig, alias="idle_gaze"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conf_version": Description(en="Configuration version", zh="配置文件版本"),
        "host": Description(en="Server host address", zh="服务器主机地址"),
        "port": Description(en="Server port number", zh="服务器端口号"),
        "config_alts_dir": Description(
            en="Directory for alternative configurations", zh="备用配置目录"
        ),
        "tool_prompts": Description(
            en="Tool prompts to be inserted into persona prompt",
            zh="要插入到角色提示词中的工具提示词",
        ),
        "enable_proxy": Description(
            en="Enable proxy mode for multiple clients",
            zh="启用代理模式以支持多个客户端使用一个 ws 连接",
        ),
        "enable_screen_capture": Description(
            en="Allow all captured images and system audio. When disabled, explicit image uploads remain allowed.",
            zh="允许所有捕获图像和系统音频。禁用时仍允许手动上传的图像。",
        ),
        "enable_screen_sharing": Description(
            en="Allow shared-screen images when enable_screen_capture is disabled, without allowing camera images or system audio.",
            zh="在 enable_screen_capture 禁用时允许共享屏幕图像，但不允许摄像头图像或系统音频。",
        ),
        "analyze_audio_url": Description(
            en="URL of the analyze-audio server (Demucs/pyannote/SER/BGM/Whisper). Used for system audio analysis.",
            zh="analyze-audio 服务器 URL (Demucs/pyannote/SER/BGM/Whisper)。用于系统音频分析。",
        ),
        "analyze_audio_fallback_urls": Description(
            en="Fallback analyze-audio servers, tried in order when the primary is unreachable.",
            zh="备用 analyze-audio 服务器，主服务器不可用时按顺序尝试。",
        ),
    }

    @model_validator(mode="after")
    def check_port(cls, values):
        port = values.port
        if port < 0 or port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        return values
