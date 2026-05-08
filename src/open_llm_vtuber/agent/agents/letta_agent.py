import asyncio
import json
import re
import urllib.parse
import urllib.request
from typing import AsyncIterator, Dict, Any, List, Optional, Union

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

_SEP = "=" * 60

# ---------------------------------------------------------------------------
# 유튜브 링크 전처리: [유튜브 링크]...[/유튜브 링크] 안의 검색 결과 URL을
# 실제 영상 URL로 교체해서 letta(claude -p)에 깔끔하게 전달
# ---------------------------------------------------------------------------
_YOUTUBE_TAG_RE = re.compile(r"\[유튜브 링크\](.*?)\[/유튜브 링크\]", re.DOTALL)
_YT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}


def _yt_fetch(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, headers=_YT_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _yt_first_video_url(html: str) -> str:
    """검색 결과 HTML에서 첫 번째 영상 URL을 반환."""
    m = re.search(r"var ytInitialData = ({.*?});</script>", html, re.DOTALL)
    if not m:
        return ""
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return ""
    sections = (
        data.get("contents", {})
        .get("twoColumnSearchResultsRenderer", {})
        .get("primaryContents", {})
        .get("sectionListRenderer", {})
        .get("contents", [])
    )
    for section in sections:
        for item in section.get("itemSectionRenderer", {}).get("contents", []):
            vid = item.get("videoRenderer", {}).get("videoId", "")
            if vid:
                return f"https://www.youtube.com/watch?v={vid}"
    return ""


async def _resolve_youtube_url(url: str) -> str:
    """검색 결과 URL이면 첫 번째 영상 URL로 교체, 영상 URL이면 그대로 반환."""
    try:
        if "results" in urllib.parse.urlparse(url).path:
            html = await asyncio.to_thread(_yt_fetch, url)
            resolved = _yt_first_video_url(html)
            if resolved:
                logger.info(f"[YouTube] 검색 결과 → 영상 URL: {resolved}")
                return resolved
    except Exception as e:
        logger.warning(f"[YouTube] URL 해석 실패 ({url}): {e}")
    return url


async def _preprocess_youtube_links(text: str) -> str:
    """[유튜브 링크] 태그 안의 URL을 실제 영상 URL로 교체."""
    matches = list(_YOUTUBE_TAG_RE.finditer(text))
    for m in matches:
        url = m.group(1).strip()
        resolved = await _resolve_youtube_url(url)
        text = text.replace(m.group(0), f"[유튜브 링크]{resolved}[/유튜브 링크]", 1)
    return text


# ---------------------------------------------------------------------------
# 일반 URL 분류 및 태그: [이미지 링크] / [웹 링크]
# ---------------------------------------------------------------------------
_IMG_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"})
_IMG_CONTENT_TYPES = frozenset({
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "image/svg+xml", "image/bmp", "image/tiff",
})
_ALL_URL_RE = re.compile(r"https?://[^\s\[\]<>\"']+")
_TAGGED_URL_RE = re.compile(
    r"\[(?:유튜브|이미지|웹) 링크\](https?://[^\[\]]+?)\[/(?:유튜브|이미지|웹) 링크\]"
)


def _head_content_type(url: str, timeout: int = 10) -> str:
    """HEAD 요청으로 Content-Type 반환. 실패 시 빈 문자열."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers=_YT_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.headers.get("Content-Type", "").lower()
    except Exception:
        return ""


async def _classify_url(url: str) -> str:
    """URL 콘텐츠 타입 분류: 'image' | 'web'. 타임아웃 10초."""
    ct = await asyncio.to_thread(_head_content_type, url)
    if any(img_ct in ct for img_ct in _IMG_CONTENT_TYPES):
        return "image"
    path = url.lower().split("?")[0].split("#")[0]
    if any(path.endswith(ext) for ext in _IMG_EXTENSIONS):
        return "image"
    return "web"


async def _tag_general_urls(text: str) -> str:
    """[유튜브/이미지/웹 링크] 외의 URL을 [이미지 링크] 또는 [웹 링크] 태그로 감쌈."""
    already_tagged = {m.group(1).strip() for m in _TAGGED_URL_RE.finditer(text)}
    to_process = [
        m.group(0) for m in _ALL_URL_RE.finditer(text)
        if m.group(0) not in already_tagged
    ]
    unique_urls = list(dict.fromkeys(to_process))
    if not unique_urls:
        return text

    types = await asyncio.gather(*[_classify_url(url) for url in unique_urls])
    type_map = dict(zip(unique_urls, types))

    for url in to_process:
        tag = "이미지" if type_map[url] == "image" else "웹"
        text = text.replace(url, f"[{tag} 링크]{url}[/{tag} 링크]", 1)

    return text


def _select_or_create_agent(
    base_url: str,
    system_prompt: str,
    model: str,
    embedding: str,
    context_window_limit: Optional[int] = None,
) -> str:
    """
    Connect to the Letta server, list existing agents for interactive selection,
    or auto-create one if none exist. Returns the chosen agent ID.
    After selection/creation, optionally updates the agent's system/model/embedding.
    """
    sync_client = Letta(base_url=base_url)

    try:
        agents = list(sync_client.agents.list())
    except Exception as e:
        raise RuntimeError(
            f"Letta 서버({base_url})에 연결할 수 없습니다: {e}\n"
            "Letta 서버가 실행 중인지 확인하세요."
        )

    print(f"\n{_SEP}")
    print(f"  Letta 서버: {base_url}")
    print(_SEP)

    if not agents:
        print("  등록된 에이전트가 없습니다. 새 에이전트를 생성합니다...")
        agent_id = _create_new_agent(sync_client, system_prompt, model, embedding, context_window_limit)
        # New agent is already created with current settings — no update needed.
        return agent_id

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
                agent_id = _create_new_agent(sync_client, system_prompt, model, embedding, context_window_limit)
                return agent_id
            if 1 <= choice <= len(agents):
                selected = agents[choice - 1]
                print(f"  선택됨: {selected.id}  ({selected.name})")
                print(_SEP + "\n")
                _confirm_and_update(sync_client, selected.id, system_prompt, model, embedding, context_window_limit)
                _ensure_core_memory(sync_client, selected.id)
                return selected.id
            print(f"  0 ~ {len(agents)} 사이의 번호를 입력하세요.")
        except ValueError:
            print("  숫자를 입력하세요.")


def _confirm_and_update(
    sync_client: Letta,
    agent_id: str,
    system_prompt: str,
    model: str,
    embedding: str,
    context_window_limit: Optional[int] = None,
) -> None:
    """Show the pending update and ask the user whether to apply it."""
    total_len = len(system_prompt)
    # Show head (persona) + tail (utils / expression prompt) so the user can
    # see both sections regardless of how long the persona prompt is.
    head = system_prompt[:200].replace("\n", " ")
    tail = system_prompt[-400:].replace("\n", " ") if total_len > 600 else ""

    print(f"\n{_SEP}")
    print("  에이전트 설정 업데이트 여부 확인")
    print(_SEP)
    print(f"  모델:        {model}")
    print(f"  임베딩:      {embedding}")
    if context_window_limit:
        print(f"  컨텍스트 윈도우 제한: {context_window_limit}")
    print(f"  프롬프트 길이: {total_len}자")
    print(f"  [앞부분] {head}...")
    if tail:
        print(f"  [뒷부분] ...{tail}")
    print(_SEP)
    logger.debug(f"[letta] system_prompt 전체:\n{system_prompt}")

    while True:
        answer = input("  이 설정으로 에이전트를 업데이트하시겠습니까? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            if context_window_limit is not None:
                sync_client.agents.update(
                    agent_id,
                    system=system_prompt,
                    model=model,
                    embedding=embedding,
                    context_window_limit=context_window_limit,
                )
            else:
                sync_client.agents.update(
                    agent_id,
                    system=system_prompt,
                    model=model,
                    embedding=embedding,
                )
            logger.info(f"에이전트 {agent_id} 업데이트 완료.")
            print(f"  업데이트 완료.\n{_SEP}\n")
            return
        if answer in ("n", "no", ""):
            print(f"  업데이트를 건너뜁니다.\n{_SEP}\n")
            return
        print("  y 또는 n을 입력하세요.")


_REQUIRED_TOOLS = [
    "core_memory_append",
    "core_memory_replace",
    "archival_memory_insert",
    "archival_memory_search",
]
_REQUIRED_BLOCKS = ["human", "persona"]


def _ensure_core_memory(sync_client: Letta, agent_id: str) -> None:
    """core_memory 도구와 블록이 없으면 자동으로 추가."""
    # 도구 확인 및 추가
    existing_tools = {t.name for t in sync_client.agents.tools.list(agent_id=agent_id)}
    all_tools = list(sync_client.tools.list())
    for tool_name in _REQUIRED_TOOLS:
        if tool_name not in existing_tools:
            tool = next((t for t in all_tools if t.name == tool_name), None)
            if tool:
                sync_client.agents.tools.attach(tool.id, agent_id=agent_id)
                logger.info(f"[core_memory] 도구 추가: {tool_name}")
            else:
                logger.warning(f"[core_memory] 도구를 찾을 수 없음: {tool_name}")

    # 블록 확인 및 추가
    existing_labels = {b.label for b in sync_client.agents.blocks.list(agent_id=agent_id)}
    for label in _REQUIRED_BLOCKS:
        if label not in existing_labels:
            block = sync_client.blocks.create(label=label, value="", limit=5000)
            sync_client.agents.blocks.attach(block.id, agent_id=agent_id)
            logger.info(f"[core_memory] 블록 추가: {label}")


def _create_new_agent(
    sync_client: Letta,
    system_prompt: str,
    model: str,
    embedding: str,
    context_window_limit: Optional[int] = None,
) -> str:
    """Create a new agent and return its ID."""
    kwargs = dict(
        model=model,
        embedding=embedding,
        system=system_prompt,
        name="Open-LLM-VTuber",
        memory_blocks=[{"label": label, "value": ""} for label in _REQUIRED_BLOCKS],
        tools=_REQUIRED_TOOLS,
    )
    if context_window_limit is not None:
        kwargs["context_window_limit"] = context_window_limit

    agent = sync_client.agents.create(**kwargs)
    logger.info(f"새 Letta 에이전트 생성 완료: {agent.id}")
    print(f"  새 에이전트 생성 완료: {agent.id}")
    print(_SEP + "\n")
    return agent.id


class LettaAgent(AgentInterface):
    """
    Custom Letta class to interface with the Letta server.

    If `id` is not set in conf.yaml, connects to the Letta server at startup,
    lists available agents for interactive selection, and creates a new one if none exist.
    After selecting an existing agent, optionally updates its system prompt, model,
    and embedding to match the current conf.yaml values.
    """

    def __init__(
        self,
        live2d_model,
        id,
        system_prompt: str = "",
        model: str = "google_ai/gemini-2.5-flash",
        embedding: str = "google_ai/gemini-embedding-2",
        tts_preprocessor_config: Optional[TTSPreprocessorConfig] = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        host: str = "localhost",
        port: int = 8283,
        context_window_limit: Optional[int] = None,
    ):
        super().__init__()
        self.url = f"http://{host}:{port}"

        if not id:
            id = _select_or_create_agent(self.url, system_prompt, model, embedding, context_window_limit)

        self.client = AsyncLetta(base_url=self.url, timeout=600)
        self.id = id
        logger.info(f"LettaAgent 초기화 완료. agent_id={self.id}, url={self.url}")

        # Build the decorated pipeline: raw str tokens → SentenceOutput
        self._chat = tts_filter(tts_preprocessor_config)(  # type: ignore[arg-type]
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
        text = await _preprocess_youtube_links(text)
        text = await _tag_general_urls(text)

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
            background=True,  # Redis 백업: 멀티 워커에서도 LLM 1회만 실행
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
