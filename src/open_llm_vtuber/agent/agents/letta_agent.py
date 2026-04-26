from typing import AsyncIterator, Dict, Any, List, Union
from loguru import logger
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from letta_client import Letta, AsyncLetta
from letta_client.types.agents.message_create_params import InputUnionMember1
from letta_client.types.agents.text_content_param import TextContentParam
from letta_client.types.agents.image_content_param import (
    ImageContentParam,
    SourceBase64Image,
)


def _select_or_create_agent(base_url: str, system_prompt: str) -> str:
    """
    Connect to the Letta server, list existing agents for user selection,
    or create a new one if none exist. Returns the chosen agent ID.
    """
    sync_client = Letta(base_url=base_url)

    try:
        agents = list(sync_client.agents.list())
    except Exception as e:
        raise RuntimeError(
            f"Letta 서버({base_url})에 연결할 수 없습니다: {e}\n"
            "Letta 서버가 실행 중인지 확인하세요."
        )

    print("\n" + "=" * 60)
    print(f"  Letta 서버: {base_url}")
    print("=" * 60)

    if not agents:
        print("  등록된 에이전트가 없습니다. 새 에이전트를 생성합니다...")
        return _create_new_agent(sync_client, system_prompt)

    print("  사용 가능한 에이전트 목록:\n")
    print(f"  {'[0]':6} 새 에이전트 생성")
    for i, agent in enumerate(agents, start=1):
        desc = f" — {agent.description}" if agent.description else ""
        print(f"  {'[' + str(i) + ']':6} {agent.id}  {agent.name}{desc}")

    print()
    while True:
        try:
            choice = int(input("  에이전트 번호를 선택하세요: ").strip())
            if choice == 0:
                return _create_new_agent(sync_client, system_prompt)
            if 1 <= choice <= len(agents):
                selected = agents[choice - 1]
                print(f"  선택됨: {selected.id}  ({selected.name})")
                print("=" * 60 + "\n")
                return selected.id
            print(f"  0 ~ {len(agents)} 사이의 번호를 입력하세요.")
        except ValueError:
            print("  숫자를 입력하세요.")


def _create_new_agent(sync_client: Letta, system_prompt: str) -> str:
    """Create a new agent with the given system prompt and return its ID."""
    agent = sync_client.agents.create(
        model="google_ai/gemini-2.5-flash",
        embedding="google_ai/gemini-embedding-2",
        system=system_prompt,
        name="Open-LLM-VTuber",
    )
    logger.info(f"새 Letta 에이전트 생성 완료: {agent.id}")
    print(f"  새 에이전트 생성 완료: {agent.id}")
    print("=" * 60 + "\n")
    return agent.id


class LettaAgent(AgentInterface):
    """
    Custom Letta class to interface with the Letta server.

    If `id` is not provided in conf.yaml, connects to the Letta server at startup,
    lists available agents for interactive selection, and creates a new one if none exist.
    """

    def __init__(
        self,
        live2d_model,
        id,
        system_prompt: str = "",
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        host: str = "localhost",
        port: int = 8283,
    ):
        super().__init__()
        self.url = f"http://{host}:{port}"

        if not id:
            id = _select_or_create_agent(self.url, system_prompt)

        self.client = AsyncLetta(base_url=self.url)
        self.id = id
        logger.info(f"LettaAgent 초기화 완료. agent_id={self.id}, url={self.url}")

        # Build the decorated pipeline around the raw token generator.
        # The decorator chain transforms raw str tokens → SentenceOutput.
        self._chat = tts_filter(tts_preprocessor_config)(
            display_processor()(
                actions_extractor(live2d_model)(
                    sentence_divider(
                        faster_first_response=faster_first_response,
                        segment_method=segment_method,
                        valid_tags=["think"],
                    )(self._chat_raw)
                )
            )
        )

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        # The Letta Server automatically stores historical messages, so this part is not needed
        pass

    def handle_interrupt(self, heard_response: str) -> None:
        pass

    async def chat(  # type: ignore[override]
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        async for item in self._chat(input_data):
            yield item

    async def _chat_raw(self, input_data: BatchInput) -> AsyncIterator[str]:
        """Raw token generator. Decorated into self._chat during __init__."""
        input_content: Union[str, List[InputUnionMember1]]
        text = self._to_text_prompt(input_data)

        if input_data.images:
            parts: List[InputUnionMember1] = [TextContentParam(type="text", text=text)]
            for img in input_data.images:
                img_data = img.data
                if img_data.startswith("data:") and "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                parts.append(
                    ImageContentParam(
                        type="image",
                        source=SourceBase64Image(
                            type="base64",
                            data=img_data,
                            media_type=img.mime_type,
                        ),
                    )
                )
            input_content = parts
        else:
            input_content = text

        stream = await self.client.agents.messages.create(
            agent_id=self.id,
            streaming=True,  # type: ignore[call-overload]
            stream_tokens=True,
            input=input_content,
        )

        async for token in stream:
            if token.message_type == "reasoning_message":
                continue
            elif token.message_type == "assistant_message":
                content = token.content
                if isinstance(content, list):
                    text = "".join(part.text for part in content)
                else:
                    text = content
            else:
                continue

            yield text

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(f"[Clipboard content: {text_data.content}]")

        return "\n".join(message_parts)
