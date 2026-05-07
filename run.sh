#!/bin/bash

pushd frontend
npm run build:web
popd

# ahttp_client 1.0.5 버그 패치: pydantic 2.12+에서 by_alias/by_name=False로 model_validate
# 호출 시 에러 발생. None으로 바꿔서 pydantic이 모델 config 기준으로 동작하게 함.
PATCH_FILE=$(uv run python -c "import ahttp_client.extension.pydantic as m; import inspect; print(inspect.getfile(m))" 2>/dev/null)
if [ -n "$PATCH_FILE" ]; then
    sed -i '' 's/by_alias: Optional\[bool\] = False,/by_alias: Optional[bool] = None,/g' "$PATCH_FILE"
    sed -i '' 's/by_name: Optional\[bool\] = False,/by_name: Optional[bool] = None,/g' "$PATCH_FILE"
fi

# ── Letta 에이전트 자동 선택/생성/업데이트 ──────────────────────────────────────
uv run python3 - <<'PYEOF'
import sys, os, re, json, socket
import yaml
from letta_client import Letta

AGENT_NAME      = "Open-LLM-VTuber"
REQUIRED_TOOLS  = [
    "core_memory_append", "core_memory_replace",
    "archival_memory_insert", "archival_memory_search",
]
REQUIRED_BLOCKS = ["human", "persona"]
SKIP_PROMPTS    = {"group_conversation_prompt", "proactive_speak_prompt", "mcp_prompt"}

# ── conf.yaml 로드 ────────────────────────────────────────────────────────────
conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf.yaml")
with open(conf_path) as f:
    conf = yaml.safe_load(f)

char_cfg  = conf.get("character_config") or {}
agent_cfg = char_cfg.get("agent_config") or {}
if agent_cfg.get("conversation_agent_choice") != "letta_agent":
    print("[Letta] letta_agent를 사용하지 않으므로 에이전트 설정 건너뜀.")
    sys.exit(0)

la       = agent_cfg.get("agent_settings", {}).get("letta_agent", {})
host     = la.get("host", "localhost")
port     = la.get("port", 8283)
model    = la.get("model", "claude_cli/claude-opus-4-6")
emb      = la.get("embedding", "ollama/bge-m3:latest")
ctx_lim  = la.get("context_window_limit", 40000)
base_url = f"http://{host}:{port}"

# ── Letta 서버 연결 확인 (포트 열림 여부만 체크) ──────────────────────────────
import socket
try:
    s = socket.create_connection((host, port), timeout=3)
    s.close()
except Exception:
    print(f"[Letta] 서버({base_url})에 연결할 수 없습니다. 에이전트 설정 건너뜀.")
    sys.exit(0)

# ── 시스템 프롬프트 조립 ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    from prompts import prompt_loader
    from open_llm_vtuber.live2d_model import Live2dModel

    persona_prompt    = char_cfg.get("persona_prompt", "")
    live2d_model_name = char_cfg.get("live2d_model_name", "")
    tool_prompts      = (conf.get("system_config") or {}).get("tool_prompts", {})

    # live2d emo_str 추출
    emo_str = ""
    if live2d_model_name:
        try:
            l2d = Live2dModel(live2d_model_name)
            l2d.set_model(live2d_model_name)
            emo_str = l2d.emo_str
        except Exception as e:
            print(f"[Letta] live2d 모델 로드 실패 ({e}), emo_str 생략")

    system_prompt = persona_prompt
    for prompt_name, prompt_file in tool_prompts.items():
        if prompt_name in SKIP_PROMPTS:
            continue
        try:
            content = prompt_loader.load_util(prompt_file)
            if prompt_name == "live2d_expression_prompt" and emo_str:
                content = content.replace("[<insert_emomap_keys>]", emo_str)
            system_prompt += content
        except Exception as e:
            print(f"[Letta] 프롬프트 로드 실패 ({prompt_name}): {e}")

    print(f"[Letta] 시스템 프롬프트 조립 완료 ({len(system_prompt)}자)")
except Exception as e:
    print(f"[Letta] 시스템 프롬프트 조립 실패: {e} — system 필드 업데이트 건너뜀")
    system_prompt = None

# ── 에이전트 찾기 또는 생성 ───────────────────────────────────────────────────
client = Letta(base_url=base_url)
agents = list(client.agents.list())
agent  = next((a for a in agents if a.name == AGENT_NAME), None)

update_kwargs = dict(model=model, embedding=emb, context_window_limit=ctx_lim)
if system_prompt is not None:
    update_kwargs["system"] = system_prompt

if agent:
    agent_id = agent.id
    print(f"[Letta] 기존 에이전트 발견: {agent_id}")
    client.agents.update(agent_id, **update_kwargs)
    print(f"[Letta] model/embedding/context_window/system 업데이트 완료")
else:
    create_kwargs = dict(
        name=AGENT_NAME,
        model=model,
        embedding=emb,
        context_window_limit=ctx_lim,
        memory_blocks=[{"label": l, "value": ""} for l in REQUIRED_BLOCKS],
        tools=REQUIRED_TOOLS,
    )
    if system_prompt is not None:
        create_kwargs["system"] = system_prompt
    agent    = client.agents.create(**create_kwargs)
    agent_id = agent.id
    print(f"[Letta] 새 에이전트 생성 완료: {agent_id}")

# ── 필수 툴/블록 확인 및 추가 ─────────────────────────────────────────────────
existing_tools = {t.name for t in client.agents.tools.list(agent_id=agent_id)}
all_tools      = list(client.tools.list())
for tool_name in REQUIRED_TOOLS:
    if tool_name not in existing_tools:
        tool = next((t for t in all_tools if t.name == tool_name), None)
        if tool:
            client.agents.tools.attach(tool.id, agent_id=agent_id)
            print(f"[Letta] 툴 추가: {tool_name}")

existing_labels = {b.label for b in client.agents.blocks.list(agent_id=agent_id)}
for label in REQUIRED_BLOCKS:
    if label not in existing_labels:
        block = client.blocks.create(label=label, value="", limit=5000)
        client.agents.blocks.attach(block.id, agent_id=agent_id)
        print(f"[Letta] 블록 추가: {label}")

# ── conf.yaml의 letta_agent.id 필드 업데이트 ─────────────────────────────────
with open(conf_path) as f:
    lines = f.readlines()

in_letta = False
updated  = False
for i, line in enumerate(lines):
    if re.search(r"^\s+letta_agent:", line):
        in_letta = True
    if in_letta and re.match(r"\s+id:\s*'[^']*'", line):
        m       = re.search(r"(#.*)", line)
        comment = f"  {m.group(1)}" if m else ""
        indent  = len(line) - len(line.lstrip())
        lines[i] = f"{' ' * indent}id: '{agent_id}'{comment}\n"
        updated  = True
        break

with open(conf_path, "w") as f:
    f.writelines(lines)

if updated:
    print(f"[Letta] conf.yaml 업데이트 완료 (id={agent_id})")
else:
    print(f"[Letta] WARNING: conf.yaml에서 id 필드를 찾지 못했습니다.")
PYEOF

uv run run_server.py