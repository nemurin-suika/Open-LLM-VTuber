# config_manager/system.py
from pydantic import Field, model_validator, BaseModel
from typing import Dict, ClassVar
from .i18n import I18nMixin, Description


class WindEffectConfig(BaseModel):
    """Wind effect configuration for Live2D physics (hair/cloth flutter)."""

    enabled: bool = Field(True, alias="enabled")
    max_strength: float = Field(0.5, alias="max_strength")
    transition_speed: float = Field(0.5, alias="transition_speed")
    interval_min: float = Field(3.0, alias="interval_min")
    interval_max: float = Field(8.0, alias="interval_max")

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
    wind_effect: WindEffectConfig = Field(
        default_factory=WindEffectConfig, alias="wind_effect"
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
            en="Whether to pass screenshot images to the LLM. Disable for text-only LLMs (e.g. Claude CLI).",
            zh="是否将屏幕截图传递给LLM。纯文本LLM（如Claude CLI）请禁用。",
        ),
    }

    @model_validator(mode="after")
    def check_port(cls, values):
        port = values.port
        if port < 0 or port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        return values
