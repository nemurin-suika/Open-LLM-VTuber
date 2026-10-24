"""세션별 보조 대화 로그 (TODO19 5단계).

letta conversation 은 "대화 원본"이지만 두 가지를 담지 못한다.

1. 네무린 발화는 턴이 **끝난 뒤에** letta 에 기록된다. 그래서 작업을 시켜 두고
   다른 세션에 갔다 돌아오면, 아직 응답이 안 끝난 턴의 내 질문이 화면에서 사라진다.
2. ``[알림: ...]`` 중간 진행 표시는 애초에 letta 에 저장되지 않는다(저장하면 다음 턴
   컨텍스트가 알림 텍스트로 오염된다). 그래서 새로고침이나 세션 이동 후에는
   중간 과정이 통째로 날아간다.

이 모듈은 그 두 가지만 세션별 JSONL 로 따로 남겨서, 히스토리 복원 때 letta 기록에
덧대어 준다. letta 대화 컨텍스트는 전혀 건드리지 않는다.

파일 경로: ``~/.mizuki/session_log/<sanitized conversation id>.jsonl``
한 줄 = ``{"ts": ISO8601(UTC), "role": "human"|"ai", "type": "text"|"progress_update",
"content": str}``
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# 로그 디렉터리. pgdata 와 같은 ~/.mizuki 아래에 모아둔다.
_LOG_DIR = os.path.join(os.path.expanduser("~"), ".mizuki", "session_log")

# 파일이 이보다 길어지면 뒤쪽만 남기고 잘라낸다(세션 하나당 상한).
_PRUNE_AT_LINES = 3000
_PRUNE_KEEP_LINES = 1200

# 복원 때 끌어올 보조 로그 상한 (letta 창 밖의 옛 알림까지 올라오지 않게).
_MERGE_MAX_ENTRIES = 300

# letta 가 턴 종료 시점에 발화를 저장하므로, 보조 로그의 턴 시작 시각은
# letta 쪽 첫 메시지 시각보다 과거다. 그 간격만큼 여유를 두고 끌어온다.
_MERGE_SLACK = timedelta(minutes=45)


def _sanitize(session_id: Optional[str]) -> str:
    """conversation id → 파일명으로 쓸 안전한 키. letta 쪽 규칙과 같게 맞춘다."""
    try:
        from .agent.agents.letta_agent import _sanitize_session_key

        return _sanitize_session_key(session_id or "default")
    except Exception:
        key = str(session_id or "default")
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in key) or "default"


def log_path(session_id: Optional[str]) -> str:
    """세션 보조 로그 파일 경로."""
    return os.path.join(_LOG_DIR, f"{_sanitize(session_id)}.jsonl")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(session_id: Optional[str], entry: Dict[str, Any]) -> None:
    path = log_path(session_id)
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:  # 로그 실패가 대화를 막아선 안 된다
        logger.debug(f"[session-log] 기록 실패 ({path}): {e}")
        return
    _maybe_prune(path)


def _maybe_prune(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= _PRUNE_AT_LINES:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[-_PRUNE_KEEP_LINES:])
        logger.debug(f"[session-log] {path} 정리: {len(lines)} → {_PRUNE_KEEP_LINES}줄")
    except Exception as e:
        logger.debug(f"[session-log] 정리 실패 ({path}): {e}")


def log_user_message(session_id: Optional[str], text: Optional[str]) -> None:
    """네무린 발화를 받은 즉시 기록 (턴이 끝나기 전에도 화면에 복원되도록)."""
    body = (text or "").strip()
    if not body:
        return
    _append(
        session_id,
        {"ts": _utc_now_iso(), "role": "human", "type": "text", "content": body},
    )


def log_progress(
    session_id: Optional[str], text: Optional[str], ts: Optional[float] = None
) -> None:
    """``[알림: ...]`` 중간 진행 표시를 기록."""
    body = (text or "").strip()
    if not body:
        return
    when = _utc_now_iso()
    if ts:
        try:
            when = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except Exception:
            pass
    _append(
        session_id,
        {"ts": when, "role": "ai", "type": "progress_update", "content": body},
    )


def drop_session_log(session_id: Optional[str]) -> None:
    """세션 삭제 시 보조 로그도 함께 지운다."""
    path = log_path(session_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.debug(f"[session-log] 삭제 실패 ({path}): {e}")


def _read_entries(session_id: Optional[str]) -> List[Dict[str, Any]]:
    path = log_path(session_id)
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict) and item.get("content"):
                    out.append(item)
    except Exception as e:
        logger.debug(f"[session-log] 읽기 실패 ({path}): {e}")
        return []
    return out[-_MERGE_MAX_ENTRIES:]


def _parse_ts(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def merge_history(
    session_id: Optional[str], messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """letta 복원 목록에 보조 로그를 끼워 넣어 시간순으로 돌려준다.

    규칙
      - 중간 알림은 letta 에 없으므로 전부 보조 로그에서 가져온다.
      - 네무린 발화는 양쪽에 다 있을 수 있다. 같은 내용이 짝지어지면 보조 로그 쪽을
        남긴다(기록 시각이 실제 발화 시각이라, 중간 알림보다 앞에 놓인다).
      - letta 쪽 가장 오래된 메시지보다 한참 과거인 로그는 버린다.
    """
    entries = _read_entries(session_id)
    if not entries:
        return messages

    letta_items: List[Dict[str, Any]] = []
    last_dt: Optional[datetime] = None
    for msg in messages:
        dt = _parse_ts(msg.get("timestamp")) or last_dt
        last_dt = dt or last_dt
        letta_items.append({"msg": dict(msg), "dt": dt})

    oldest_dt = next((it["dt"] for it in letta_items if it["dt"]), None)
    cutoff = (oldest_dt - _MERGE_SLACK) if oldest_dt else None

    log_items: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        dt = _parse_ts(entry.get("ts"))
        if cutoff and dt and dt < cutoff:
            continue
        log_items.append(
            {
                "msg": {
                    "id": f"slog-{entry.get('ts') or idx}-{idx}",
                    "role": entry.get("role") or "ai",
                    "type": entry.get("type") or "text",
                    "content": str(entry.get("content") or ""),
                    "timestamp": str(entry.get("ts") or ""),
                },
                "dt": dt,
                "entry": entry,
            }
        )

    if not log_items:
        return messages

    # 같은 네무린 발화가 양쪽에 있으면 letta 쪽을 뺀다 (앞에서부터 1:1 로 짝짓기).
    pending_user = [
        it for it in log_items if it["entry"].get("role") == "human"
    ]
    for it in letta_items:
        msg = it["msg"]
        if msg.get("role") != "human":
            continue
        body = str(msg.get("content") or "").strip()
        for cand in pending_user:
            if cand.get("used"):
                continue
            if str(cand["entry"].get("content") or "").strip() == body:
                cand["used"] = True
                it["drop"] = True
                break

    merged = [it for it in letta_items if not it.get("drop")] + log_items
    base = oldest_dt or datetime.now(timezone.utc)
    for pos, it in enumerate(merged):
        it["pos"] = pos
    merged.sort(key=lambda it: (it["dt"] or base, it["pos"]))

    out: List[Dict[str, Any]] = []
    for it in merged:
        msg = it["msg"]
        msg.setdefault("type", "text")
        out.append(msg)
    logger.info(
        f"[session-log] 복원 병합: letta {len(messages)}건 + 보조 {len(log_items)}건 "
        f"→ {len(out)}건 (conv={session_id})"
    )
    return out
