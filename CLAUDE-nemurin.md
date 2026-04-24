# CLAUDE.md

이 파일은 이 저장소에서 작업할 때 Claude Code(claude.ai/code)에게 안내를 제공합니다.

## 프로젝트 개요

Open-LLM-VTuber는 Live2D 아바타를 지원하는 음성 대화형 AI 컴패니언입니다. FastAPI WebSocket 서버, 모듈식 AI 엔진(LLM, ASR, TTS, VAD), 프론트엔드 서브모듈로 구성된 Python 애플리케이션(Python 3.10–3.12 필요)입니다.

## 주요 명령어

```bash
uv sync                          # 의존성 설치
uv run run_server.py             # 서버 시작
uv run run_server.py --verbose   # 상세 로그 모드로 시작
uv run upgrade.py                # 프로젝트 업그레이드
ruff check .                     # 린트 검사
ruff format .                    # 코드 포맷
pre-commit run --all-files       # 모든 pre-commit 훅 실행
```

자동화된 테스트는 없으며, 웹 인터페이스를 통해 수동으로 테스트합니다.

## 설정

- 사용자 설정: `conf.yaml` (`config_templates/conf.default.yaml`에서 복사)
- 캐릭터 설정: `characters/` 디렉토리 (YAML, 런타임에 기본 설정 위에 병합됨)
- 런타임 설정 전환: 클라이언트가 파일명과 함께 `switch-config` WebSocket 메시지를 전송
- `service_context.py`의 `deep_merge()`가 캐릭터 YAML을 기본 설정 위에 병합
- 설정 스키마를 변경할 경우 `config_templates/conf.default.yaml`과 `config_templates/conf.ZH.default.yaml` 모두 업데이트 필요

## 아키텍처

### 요청 흐름

```
브라우저/클라이언트
  → WebSocket /client-ws
  → WebSocketHandler (websocket_handler.py)
  → MessageType 디스패치 → _handle_* 메서드
  → handle_conversation_trigger() (conversations/conversation_handler.py)
  → process_single_conversation() 또는 process_group_conversation()
  → AgentInterface.chat() → SentenceOutput / AudioOutput yield
  → TTSTaskManager.speak() → WebSocket으로 오디오 반환
```

### 핵심 컴포넌트

**`service_context.py` — `ServiceContext`**: 모든 엔진 인스턴스(ASR, TTS, VAD, 에이전트, 번역, MCP)를 보유하는 연결별 중앙 컨테이너. WebSocket 연결마다 각자의 인스턴스를 가집니다. `load_from_config()`는 YAML에서 초기화하고, `load_cache()`는 공유 연결을 위해 사전 생성된 인스턴스를 참조로 받습니다.

**`websocket_handler.py` — `WebSocketHandler`**: `MessageType` enum과 `_init_message_handlers()` 딕셔너리를 통해 수신 WebSocket 메시지를 라우팅합니다. `client_connections`, `client_contexts`, `chat_group_manager`, `current_conversation_tasks`를 관리합니다.

**`conversations/`**: 대화 턴을 조율합니다.
- `conversation_handler.py` — 최상위 트리거 디스패치
- `single_conversation.py` — 1:1 사용자 ↔ AI 대화
- `group_conversation.py` — 다중 사용자 공유 AI 대화
- `tts_manager.py` — `TTSTaskManager`가 병렬 TTS 생성을 큐에 넣으면서 전달 순서를 유지

**`chat_group.py` — `ChatGroupManager`**: 공유 그룹 대화를 위해 여러 클라이언트 연결을 그룹으로 묶습니다. 각 클라이언트는 항상 정확히 하나의 그룹에 속합니다.

### 엔진 팩토리 패턴

모든 AI 엔진은 `*_interface.py` → 구체적 구현체 → `*_factory.py` → `ServiceContext.init_*()` 순서를 따릅니다.

설정이 변경되지 않으면 재초기화를 건너뜁니다(생성 전 동등성 검사):

| 엔진 | 팩토리 | 인터페이스 | 설정 키 |
|------|--------|-----------|---------|
| ASR | `asr/asr_factory.py` | `ASRInterface` | `asr_config.asr_model` |
| TTS | `tts/tts_factory.py` | `TTSInterface` | `tts_config.tts_model` |
| VAD | `vad/vad_factory.py` | `VADInterface` | `vad_config.vad_model` |
| Agent | `agent/agent_factory.py` | `AgentInterface` | `agent_config.conversation_agent_choice` |
| Translate | `translate/translate_factory.py` | `TranslateInterface` | `translator_config.translate_provider` |

### 에이전트 시스템

`AgentFactory.create_agent()`는 `basic_memory_agent`, `mem0_agent`, `hume_ai_agent`, `letta_agent`를 지원합니다.

`basic_memory_agent`(기본값)는 `LLMFactory`로부터 `StatelessLLMInterface`를 조합합니다:
- `openai_compatible_llm`: openai, gemini, zhipu, deepseek, groq, mistral, lmstudio 처리
- `ollama_llm`, `claude_llm`, `llama_cpp_llm`, `stateless_llm_with_template`

`AgentInterface.chat()`은 `BatchInput`(텍스트 + 선택적 이미지/파일 + 메타데이터)을 받아 `SentenceOutput` 또는 `AudioOutput`을 yield합니다.

### MCP 연동

`mcpp/`는 Model Context Protocol 도구 사용을 처리합니다. MCP는 `basic_memory_agent`만 지원합니다. `ServiceContext`에서의 초기화 순서: `ServerRegistry` → `ToolAdapter.get_tools()` → `ToolManager` → `MCPClient` → `ToolExecutor`. MCP 프롬프트 문자열은 에이전트의 시스템 프롬프트에 주입됩니다.

### WebSocket 메시지 타입 (MessageType enum)

| 카테고리 | 메시지 |
|---------|--------|
| CONVERSATION | `mic-audio-end`, `text-input`, `ai-speak-signal` |
| CONTROL | `interrupt-signal`, `audio-play-start` |
| DATA | `mic-audio-data` |
| CONFIG | `fetch-configs`, `switch-config` |
| HISTORY | `fetch-history-list`, `fetch-and-set-history`, `create-new-history`, `delete-history` |
| GROUP | `add-client-to-group`, `remove-client-from-group` |

### REST/WebSocket 엔드포인트 (routes.py)

- `WS /client-ws` — 메인 클라이언트 연결
- `WS /proxy-ws` — 다른 서버 인스턴스로의 프록시
- `WS /tts-ws` — 독립 TTS 생성
- `POST /asr` — 독립 오디오 전사
- `GET /live2d-models/info` — 사용 가능한 모델 목록 조회

## 주요 개발 패턴

### 새 엔진 추가

1. 해당 디렉토리에 `*_interface.py` 구현
2. 팩토리의 if/elif 체인에 추가
3. `config_manager/`에 Pydantic 설정 클래스 추가
4. `ServiceContext.init_*()`에 등록
5. `config_templates/` YAML 파일 두 개 모두 업데이트

### 새 WebSocket 메시지 타입 추가

1. `websocket_handler.py`의 `MessageType` enum에 추가
2. `WebSocketHandler`에 `_handle_*` 메서드 추가
3. `_init_message_handlers()`에 등록

### 시스템 프롬프트 구성

`ServiceContext.construct_system_prompt()`는 `system_config.tool_prompts`의 도구 프롬프트를 페르소나 프롬프트에 추가합니다. Live2D 표정 키는 `[<insert_emomap_keys>]` 플레이스홀더를 통해 주입됩니다. MCP 프롬프트는 에이전트가 별도로 추가합니다.

### Live2D 모델

각 모델은 `live2d-models/<name>/<name>.model3.json`에 위치합니다. `Live2dModel`은 표정 이름을 쉼표로 구분된 `emo_str`로 파싱하여 시스템 프롬프트에 주입합니다. 표정과 모션은 `SentenceOutput`/`AudioOutput`의 `Actions`의 일부로 프론트엔드에 전송됩니다.

## 주요 파일 위치

| 용도 | 경로 |
|------|------|
| 진입점 | `run_server.py` |
| FastAPI 서버 | `src/open_llm_vtuber/server.py` |
| WebSocket 라우팅 | `src/open_llm_vtuber/routes.py` |
| 메시지 디스패치 | `src/open_llm_vtuber/websocket_handler.py` |
| 엔진 컨테이너 | `src/open_llm_vtuber/service_context.py` |
| 대화 로직 | `src/open_llm_vtuber/conversations/` |
| 설정 클래스 | `src/open_llm_vtuber/config_manager/` |
| 에이전트 I/O 타입 | `src/open_llm_vtuber/agent/input_types.py`, `output_types.py` |
| 프롬프트 | `prompts/` |
| 대화 기록 | `chat_history/` |
| 오디오 캐시 | `cache/` |

## 패키지 관리

**uv**를 사용합니다(pip 아님). 의존성은 `pyproject.toml`에 정의되어 있습니다. Bilibili 라이브 스트리밍 지원을 위한 선택적 `bilibili` 엑스트라가 있습니다(`scripts/run_bilibili_live.py`). Windows/Linux에서 CUDA를 사용할 경우 uv 대신 **pixi**를 사용하세요(conda-forge를 통해 CUDA/cuDNN을 처리합니다).
