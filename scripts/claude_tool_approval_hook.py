#!/usr/bin/env python3
"""Claude Code PreToolUse hook — VTuber 프론트로 도구 사용 승인을 중계.

letta가 띄운 `claude -p` 가 도구를 쓰려 할 때마다 이 hook이 호출된다.

판정 순서:
  1) SAFE_AUTO 도구(WebFetch, 스크린샷 Read)는 무조건 허용 — 프론트에 묻지 않음.
  2) 영구 allowlist("항상 승인")에 살아있는(미만료) 규칙이 있으면 허용.
  3) 그 외에는 VTuber 백엔드 /tool-approval 로 POST → 네무린의 결정을 기다림.
     - "항상 승인" 응답이면 규칙을 allowlist에 TTL(기본 3일)과 함께 저장.
  4) 백엔드 미연결/타임아웃/오류는 모두 fail-closed(거부).

stdlib만 사용한다 (letta venv와 무관한 system python3에서도 동작).

환경변수:
  VTUBER_APPROVAL_URL        기본 http://127.0.0.1:12393/tool-approval
  VTUBER_CLAUDE_ALLOWLIST    기본 ~/.config/open-llm-vtuber/claude_tool_allowlist.json
  VTUBER_APPROVAL_TTL_DAYS   기본 3
  VTUBER_APPROVAL_HTTP_TIMEOUT  기본 310 (초)
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

APPROVAL_URL = os.environ.get("VTUBER_APPROVAL_URL", "http://127.0.0.1:12393/tool-approval")
ALLOWLIST_PATH = os.path.expanduser(
    os.environ.get("VTUBER_CLAUDE_ALLOWLIST", "~/.config/open-llm-vtuber/claude_tool_allowlist.json")
)
TTL_DAYS = float(os.environ.get("VTUBER_APPROVAL_TTL_DAYS", "3"))
HTTP_TIMEOUT = float(os.environ.get("VTUBER_APPROVAL_HTTP_TIMEOUT", "310"))

# letta_agent.py가 tool_call 직전에 reasoning을 여기에 저장
_INTENT_FILE = os.environ.get("MIZUKI_INTENT_FILE", "/tmp/mizuki_last_reasoning.json")
_INTENT_MAX_AGE = 30.0  # 30초 이상 지난 reasoning은 stale로 무시


def _extract_intent_from_text(text: str) -> str:
    """텍스트에서 툴 승인 팝업용 intent 추출.

    우선순위:
    1. {의도} 패턴 — 마지막 중괄호 안 텍스트 (툴 승인 팝업 전용 포맷)
    2. [알림: ...] 패턴 — 마지막 알림 (fallback, {} 없을 때만)
    """
    # 1차: {의도} 패턴
    matches = list(re.finditer(r'\{([^{}]+)\}', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    # 2차: [알림: ...] fallback
    matches = list(re.finditer(r'\[알림:\s*(.*?)\]', text, re.DOTALL))
    return matches[-1].group(1).strip() if matches else ""


def _read_intent_from_transcript(transcript_path: str) -> str:
    """Claude Code transcript JSONL에서 최근 assistant 메시지의 intent 텍스트 추출.
    직접 Claude Code 실행 시 letta_agent가 파일을 갱신하지 않을 때의 fallback.

    추출 우선순위: {의도} → [알림: ...] (둘 다 없으면 빈 문자열)

    Claude Code transcript JSONL 실제 형식:
      {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "..."}]}, ...}
    """
    if not transcript_path:
        return ""
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = [l for l in f.readlines() if l.strip()]
        # 최근 50줄에서 역순으로 가장 가까운 intent 검색
        for line in reversed(lines[-50:]):
            try:
                msg = json.loads(line.strip())
            except Exception:
                continue
            # Claude Code transcript: type == "assistant" (role 필드가 아닌 type 필드 사용)
            is_assistant = (
                msg.get("type") == "assistant"
                or msg.get("role") == "assistant"
            )
            if not is_assistant:
                continue
            # message 하위 구조에서 content 추출
            inner = msg.get("message", msg)
            content = inner.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        intent = _extract_intent_from_text(part.get("text", ""))
                        if intent:
                            return intent
            elif isinstance(content, str):
                intent = _extract_intent_from_text(content)
                if intent:
                    return intent
    except Exception:
        pass
    return ""


def _read_last_intent(transcript_path: str = "") -> str:
    """미즈키의 최신 reasoning/알림 텍스트를 반환.

    1차: letta_agent가 기록한 파일 (30초 이내)
    2차: Claude Code transcript에서 [알림:...] 파싱 (직접 실행 시 fallback)
    """
    # 1차: 파일 방식 (letta 경유)
    try:
        with open(_INTENT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - float(data.get("ts", 0)) < _INTENT_MAX_AGE:
            return str(data.get("text", "")).strip()
    except Exception:
        pass
    # 2차: transcript fallback (직접 Claude Code 실행)
    return _read_intent_from_transcript(transcript_path)

# 기존 자동 도구 — letta claude_cli_client가 원래 --allowedTools로 허용하던 것들.
# 대부분 읽기 전용이라 네무린에게 묻지 않고 무조건 승인.
_SCREENSHOT_PREFIXES = ("/tmp/claude_screenshot", "/tmp/claude_yt_frame")


def _emit(decision: str, reason: str = "") -> None:
    """claude에게 PreToolUse 결정을 출력하고 종료."""
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,  # "allow" | "deny"
        }
    }
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(0)


def _is_safe_auto(tool_name: str, tool_input: dict) -> bool:
    if tool_name == "WebFetch":
        return True
    if tool_name == "Read":
        fp = str(tool_input.get("file_path", ""))
        return fp.startswith(_SCREENSHOT_PREFIXES)
    return False


# 선행 'cd <dir> &&' 또는 'cd <dir> ;' 한 토막을 잡는 패턴.
# 미즈키가 디렉토리 이동 때문에 항상 'cd ... && 실제명령' 형태로 실행하므로,
# 이 선행부를 벗겨내야 '항상 승인'이 'cd 전체 = bash 전체'로 번지지 않는다.
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)\s*(?:&&|;)\s*")


def _strip_cd(cmd: str) -> str:
    """선행 'cd … &&/;' 체인을 반복 제거해 실제 명령만 남긴다."""
    prev = None
    while prev != cmd:
        prev = cmd
        cmd = _CD_PREFIX_RE.sub("", cmd, count=1).strip()
    return cmd


def _bash_binary(real_cmd: str) -> str:
    toks = real_cmd.split()
    return toks[0] if toks else ""


def _rule_options(tool_name: str, tool_input: dict):
    """'항상 승인' 시 고를 수 있는 규칙 후보 목록.

    각 항목: {id, label, match, value}
      - match="tool"   value=도구명         → 그 도구 전체
      - match="binary" value=실행파일명      → cd 제거 후 그 프로그램 전체 (예: git)
      - match="exact"  value=정규화된 명령   → 정확히 이 명령만
    dialog에는 id/label만 노출하고, match/value는 hook이 저장/매칭에 쓴다.
    """
    if tool_name == "Bash":
        raw = str(tool_input.get("command", "")).strip()
        real = _strip_cd(raw)
        binary = _bash_binary(real)
        opts = []
        if binary:
            opts.append(
                {"id": "binary", "label": f"{binary} 명령 전체", "match": "binary", "value": binary}
            )
        if real:
            shown = real if len(real) <= 60 else real[:57] + "…"
            opts.append(
                {"id": "exact", "label": f"이 명령만: {shown}", "match": "exact", "value": real}
            )
        if not opts:  # 빈 명령 방어
            opts.append({"id": "tool", "label": "Bash 도구 전체", "match": "tool", "value": "Bash"})
        return opts
    return [{"id": "tool", "label": f"{tool_name} 도구 전체", "match": "tool", "value": tool_name}]


def _call_matches_rule(tool_name: str, tool_input: dict, rule: dict) -> bool:
    """저장된 규칙이 이번 호출에 적용되는지 판정."""
    match = rule.get("match")
    value = rule.get("value")
    if match == "tool":
        return tool_name == value
    if tool_name != "Bash":
        return False
    real = _strip_cd(str(tool_input.get("command", "")).strip())
    if match == "binary":
        return _bash_binary(real) == value
    if match == "exact":
        return real == value
    return False


def _load_allowlist() -> dict:
    try:
        with open(ALLOWLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("rules"), list):
                return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"rules": []}


def _save_allowlist(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(ALLOWLIST_PATH), exist_ok=True)
        tmp = f"{ALLOWLIST_PATH}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ALLOWLIST_PATH)  # 원자적 교체
    except OSError:
        pass


def _allowlist_match(tool_name: str, tool_input: dict) -> bool:
    """미만료 규칙 중 이번 호출에 적용되는 게 있는지. 만료분은 정리 후 저장."""
    data = _load_allowlist()
    now = time.time()
    kept = [r for r in data["rules"] if float(r.get("expires_at", 0)) > now]
    if len(kept) != len(data["rules"]):
        _save_allowlist({"rules": kept})
    return any(_call_matches_rule(tool_name, tool_input, r) for r in kept)


def _allowlist_add(option: dict) -> None:
    """선택된 규칙 옵션(match/value/label)을 TTL과 함께 저장. 같은 match+value는 갱신."""
    data = _load_allowlist()
    now = time.time()
    rules = [
        r
        for r in data["rules"]
        if not (r.get("match") == option.get("match") and r.get("value") == option.get("value"))
        and float(r.get("expires_at", 0)) > now
    ]
    rules.append(
        {
            "match": option.get("match"),
            "value": option.get("value"),
            "label": option.get("label", ""),
            "created_at": now,
            "expires_at": now + TTL_DAYS * 86400,
        }
    )
    _save_allowlist({"rules": rules})


def _ask_backend(tool_name: str, tool_input: dict, rule_options: list, transcript_path: str = "") -> dict:
    body = json.dumps(
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
            # dialog에는 id/label만 노출
            "rule_options": [{"id": o["id"], "label": o["label"]} for o in rule_options],
            "ttl_days": TTL_DAYS,
            "intent": _read_last_intent(transcript_path),  # 미즈키의 reasoning을 승인 팝업에 표시
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        APPROVAL_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        _emit("deny", "hook 입력 파싱 실패")

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}
    # Claude Code가 직접 실행될 때 transcript_path로 [알림:...] fallback 제공
    transcript_path = payload.get("transcript_path", "")

    # 1) 안전한 기존 자동 도구
    if _is_safe_auto(tool_name, tool_input):
        _emit("allow")

    options = _rule_options(tool_name, tool_input)

    # 2) 영구 allowlist("항상 승인", 미만료)
    if _allowlist_match(tool_name, tool_input):
        _emit("allow", "이전에 '항상 승인'된 규칙에 해당 (TTL 내)")

    # 3) 백엔드 중계 → 네무린 결정 대기
    try:
        result = _ask_backend(tool_name, tool_input, options, transcript_path)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as e:
        _emit("deny", f"승인 서버 응답 없음({type(e).__name__}) — 안전상 거부")

    decision = result.get("decision", "deny")
    if decision == "allow":
        if result.get("scope") == "always":
            # 프론트가 고른 옵션 id로 저장 규칙 결정 (없으면 첫 옵션).
            chosen_id = result.get("rule_option")
            chosen = next((o for o in options if o["id"] == chosen_id), options[0])
            _allowlist_add(chosen)
        _emit("allow")
    else:
        reason = result.get("reason") or "네무린이 이 도구 사용을 거부했습니다"
        _emit("deny", reason)


if __name__ == "__main__":
    main()
