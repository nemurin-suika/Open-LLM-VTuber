import asyncio
import base64 as _base64
import glob as _glob
import json
import os
import re
import time as _time
import urllib.parse
import urllib.request
import uuid as _uuid
from contextvars import ContextVar
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
# YouTube 영상 다운로드 및 프레임/오디오 분석
# ---------------------------------------------------------------------------
_YT_DLP_PATH = os.path.expanduser("~/develop/Open-LLM-VTuber-v1.2.1-en/yt-dlp_macos")
_YT_FRAME_COUNT = 10
_YT_DOWNLOAD_TIMEOUT = 180  # 초
_ANALYZE_AUDIO_URL = os.environ.get("ANALYZE_AUDIO_URL", "http://192.168.0.15:9881")
_REASONING_FILE = "/tmp/mizuki_last_reasoning.json"
# claude_cli_client.py가 실시간으로 쓰는 중간 알림 공유 파일
_PROGRESS_NOTIFY_FILE = "/tmp/mizuki_progress_notify.jsonl"
# 인터럽트 신호: letta 서버의 claude_cli_client.py가 polling → proc.kill()
_ABORT_FLAG_FILE = "/tmp/mizuki_abort.flag"
# ── 다중 세션(letta conversation) 대응 ──────────────────────────────────────
# letta 쪽 claude_cli_client.py의 progress_notify_file/abort_flag_file과 완전히 같은 규칙.
# 두 프로세스가 같은 경로를 계산해야 알림/인터럽트가 세션별로 올바르게 전달된다.
_DEFAULT_SESSION_KEY = "default"
_SESSION_KEY_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


def _sanitize_session_key(raw: Optional[str]) -> str:
    if not raw:
        return _DEFAULT_SESSION_KEY
    key = _SESSION_KEY_UNSAFE_RE.sub("_", str(raw).strip())[:64]
    return key or _DEFAULT_SESSION_KEY


def progress_notify_file(session_key: str = _DEFAULT_SESSION_KEY) -> str:
    """세션별 중간 알림 파일 경로 (letta 측과 동일 규칙)."""
    key = _sanitize_session_key(session_key)
    if key == _DEFAULT_SESSION_KEY:
        return _PROGRESS_NOTIFY_FILE
    return f"/tmp/mizuki_progress_notify_{key}.jsonl"


def abort_flag_file(session_key: str = _DEFAULT_SESSION_KEY) -> str:
    """세션별 인터럽트 플래그 파일 경로 (letta 측과 동일 규칙)."""
    key = _sanitize_session_key(session_key)
    if key == _DEFAULT_SESSION_KEY:
        return _ABORT_FLAG_FILE
    return f"/tmp/mizuki_abort_{key}.flag"


def reasoning_file(session_key: str = _DEFAULT_SESSION_KEY) -> str:
    """세션별 툴 승인 의도 파일 경로 (letta 측 intent_file 과 동일 규칙).

    단일 파일을 공유하면 두 세션이 동시에 도는 동안 A 세션의 {의도} 가
    B 세션 승인 팝업에 뜰 수 있다. 읽는 쪽(claude_tool_approval_hook.py)은
    claude_cli_client.py 가 주입하는 MIZUKI_INTENT_FILE env 로 같은 경로를 받는다.
    """
    key = _sanitize_session_key(session_key)
    if key == _DEFAULT_SESSION_KEY:
        return _REASONING_FILE
    return f"/tmp/mizuki_last_reasoning_{key}.json"


# ── 다중 세션 동시 처리 (TODO19 3단계) ──────────────────────────────────────
# LettaAgent 인스턴스는 전체 백엔드에서 하나만 공유된다. 그래서 세션 상태를
# 인스턴스 필드에 두면 두 세션이 동시에 돌 때 서로의 값을 덮어쓴다.
#
# 해결: 세션 상태를 ContextVar 로 옮긴다. asyncio.create_task 는 생성 시점의
# context 를 복사해 task 전용으로 들고 가므로, capture task 시작 시 세션을
# 한 번 set 하면 그 task 안의 모든 호출(agent.chat 내부 깊은 곳까지)이
# 자기 세션 값만 보게 된다. 다른 task 에는 전혀 영향이 없다.
_SESSION_CTX: ContextVar[Optional[str]] = ContextVar(
    "mizuki_active_session", default=None
)
# 실시간 전송 함수(스크린샷·세션 요약)도 세션마다 다르므로 같이 task 격리한다.
_SEND_FN_CTX: ContextVar[Optional[Any]] = ContextVar(
    "mizuki_session_send_fn", default=None
)


def bind_session(conversation_id: Optional[str]) -> None:
    """현재 asyncio task 의 세션을 고정한다 (capture task 진입 시 1회 호출)."""
    _SESSION_CTX.set(conversation_id or None)


def current_bound_session() -> Optional[str]:
    """현재 task 에 바인딩된 세션 (없으면 None = UI 활성 세션을 따른다)."""
    return _SESSION_CTX.get()


# ── 세션 메타: 사용자가 정한 이름 + 미즈키가 쓰는 최근 대화 한 줄 요약 ──
# letta Conversation 스키마에는 name 필드가 없고 summary 하나뿐이라
# summary 안에 JSON {"n": 이름, "g": 한 줄 요약} 두 값을 같이 넣는다.
# 과거에 평문으로 저장된 summary 는 이름으로 간주한다(하위호환).
# letta 서버는 summary 를 create/fork 때만 쓰고 자동으로 덮어쓰지 않는다(2026-10-24 실측).
_DEFAULT_SESSION_NAME = "메인"
_DEFAULT_META_FILE = os.path.expanduser("~/.mizuki/session_default_meta.json")


def _encode_session_meta(name: str, gist: str) -> str:
    return json.dumps(
        {"n": (name or "").strip(), "g": (gist or "").strip()}, ensure_ascii=False
    )


def _decode_session_meta(summary: Optional[str]) -> tuple:
    """summary 문자열 → (이름, 요약)."""
    raw = (summary or "").strip()
    if not raw:
        return "", ""
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return (
                    str(data.get("n") or "").strip(),
                    str(data.get("g") or "").strip(),
                )
        except Exception:
            pass
    return raw, ""


def _load_default_meta() -> tuple:
    """default(메인) 세션은 letta 에 row 가 없어 메타를 로컬 파일에 보관한다."""
    try:
        with open(_DEFAULT_META_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            str(data.get("n") or _DEFAULT_SESSION_NAME),
            str(data.get("g") or ""),
        )
    except Exception:
        return _DEFAULT_SESSION_NAME, ""


def _save_default_meta(name: str, gist: str) -> None:
    try:
        os.makedirs(os.path.dirname(_DEFAULT_META_FILE), exist_ok=True)
        with open(_DEFAULT_META_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"n": name or _DEFAULT_SESSION_NAME, "g": gist or ""},
                f,
                ensure_ascii=False,
            )
    except Exception as e:
        logger.debug(f"[session] default 메타 저장 실패: {e}")


# ADB 스크린샷 감시: /tmp/*.png 전체 감시 → 변경/신규 파일 자동 전송
_ADB_SCREENSHOT_DIR = "/tmp"
# 팝업 표시 전 제거할 Live2D 태그 패턴 (표정/시선/볼륨)
_LIVE2D_TAG_RE = re.compile(
    r'\[(look_\w+|화남|역겨움|즐거움|슬픔|부끄러움|혼란함|volume_up|volume_down)\]'
)
# 미즈키 응답 안에서 프론트 채팅창에 이미지를 첨부하기 위한 태그
#   기본형: [이미지]/Users/nemulin/develop/create-illust/output/xxx.png[/이미지]
#   폴백:   <image_display path="/absolute/path.png" />  (LLM이 습관적으로 뱉는 XML 스타일)
# 백엔드가 감지하면 파일을 base64로 인코딩해 image_display 이벤트로 전송하고
# TTS/채팅 텍스트에서는 태그와 경로를 통째로 제거한다.
# 두 형식 모두 매치하도록 alternation 사용. 경로는 group(1) 또는 group(2)에 담긴다.
_MIZUKI_IMAGE_TAG_RE = re.compile(
    r'\[이미지\](.+?)\[/이미지\]'
    r'|<image_display\s+path\s*=\s*["\']?([^"\'>]+?)["\']?\s*/?>',
    re.DOTALL,
)
# 미즈키가 턴 끝에 남기는 현재 세션 한 줄 요약 태그
#   [세션요약]지금 세션에서 무슨 얘기를 하고 있는지 한 문장[/세션요약]
# 백엔드가 감지하면 활성 세션의 요약으로 저장하고 TTS/표시 텍스트에서는 제거한다.
_SESSION_GIST_TAG_RE = re.compile(r'\[세션요약\](.+?)\[/세션요약\]', re.DOTALL)
_SESSION_GIST_OPEN = "[세션요약]"
_SESSION_GIST_CLOSE = "[/세션요약]"
# 닫는 태그를 빼먹었을 때의 폴백 — 여는 태그부터 그 줄 끝까지를 요약으로 본다.
_SESSION_GIST_OPEN_ONLY_RE = re.compile(r'\[세션요약\][^\n]*')


def _gist_tag_partial_suffix_len(text: str) -> int:
    """text 꼬리가 세션요약 태그 문자열의 앞부분과 일치하면 그 길이를 반환.

    스트리밍 청크 경계에서 '[세션요'처럼 태그 이름 중간이 잘리면 태그 감지가
    실패해 요약문이 TTS/표시 텍스트로 새어나간다. 잘린 꼬리를 다음 청크까지
    보류했다가 합쳐서 판정하기 위한 길이 계산.
    """
    if not text:
        return 0
    max_len = max(len(_SESSION_GIST_OPEN), len(_SESSION_GIST_CLOSE)) - 1
    for n in range(min(max_len, len(text)), 0, -1):
        tail = text[-n:]
        if _SESSION_GIST_OPEN.startswith(tail) or _SESSION_GIST_CLOSE.startswith(tail):
            return n
    return 0


def _extract_session_gist(buffer: str, current: str = "") -> str:
    """누적 버퍼에서 세션 한 줄 요약을 뽑는다(마지막 것 채택).

    짝이 맞는 태그를 우선하고, 마지막 여는 태그가 닫히지 않았으면 그 줄 끝까지를
    요약으로 받아들인다. 못 찾으면 기존 값(current)을 그대로 돌려준다.
    """
    gist = current
    for _gm in _SESSION_GIST_TAG_RE.finditer(buffer):
        _g = _gm.group(1).strip()
        if _g:
            gist = _g
    _last_open = buffer.rfind(_SESSION_GIST_OPEN)
    if _last_open != -1 and _SESSION_GIST_CLOSE not in buffer[_last_open:]:
        _tail = buffer[_last_open + len(_SESSION_GIST_OPEN):].split("\n", 1)[0].strip()
        if _tail:
            gist = _tail
    return gist
_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# 프론트에서 올라온 이미지를 /tmp 에 떨어뜨릴 때 쓸 확장자.
# Read 도구가 확장자로 포맷을 판별하므로 mime 과 맞춰줘야 한다.
_IMAGE_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
}


def _load_local_image_as_b64(path: str) -> tuple[str, str, str]:
    """로컬 파일 경로 → (base64, mime, label). 실패 시 ("", "", "")."""
    try:
        _p = os.path.expanduser(path.strip())
        with open(_p, "rb") as _f:
            _data = _f.read()
        _ext = os.path.splitext(_p)[1].lower()
        _mime = _MIME_BY_EXT.get(_ext, "image/png")
        return _base64.b64encode(_data).decode(), _mime, os.path.basename(_p)
    except Exception as _e:
        logger.warning(f"[image_display] 로컬 이미지 읽기 실패: {path} — {_e}")
        return "", "", ""


async def _extract_audio(video_path: str, audio_path: str) -> bool:
    """ffmpeg으로 영상에서 모노 오디오 추출 (분석에 충분한 22050Hz)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "22050", audio_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000
    except Exception:
        return False


async def _analyze_audio_call(
    endpoint: str,
    audio_path: str,
    segments: Optional[list[dict]] = None,
    timeout: float = 300.0,
) -> Optional[Any]:
    """analyze-audio 서버 호출. 실패 시 None."""
    try:
        import httpx
        url = f"{_ANALYZE_AUDIO_URL.rstrip('/')}/{endpoint}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(audio_path, "rb") as f:
                files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
                data = {"segments": json.dumps(segments)} if segments is not None else None
                r = await client.post(url, files=files, data=data)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"[Audio] {endpoint} 호출 실패 (무시): {e}")
        return None


async def _aa_separate(audio_path: str, out_path: str) -> bool:
    """analyze-audio /separate → vocals 파일 저장. 성공 여부 반환."""
    try:
        import httpx
        url = f"{_ANALYZE_AUDIO_URL.rstrip('/')}/separate"
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(audio_path, "rb") as f:
                r = await client.post(
                    url,
                    files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
                    data={"stem": "vocals"},
                )
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        logger.warning(f"[Audio] /separate 실패 (무시): {e}")
        return False


async def _audio_duration(audio_path: str) -> float:
    """ffprobe로 오디오 길이(초) 반환. 실패 시 0."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", audio_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return float(out.decode().strip())
    except Exception:
        return 0.0


_AUDIO_CHUNK_SEC = 30.0


def _build_chunks(duration: float, chunk_sec: float = _AUDIO_CHUNK_SEC) -> list[dict]:
    """오디오를 chunk_sec 단위 세그먼트 리스트로 분할. 자투리 5초 미만은 흡수."""
    if duration <= chunk_sec:
        return [{"start": 0.0, "end": round(duration, 2)}]
    segs: list[dict] = []
    t = 0.0
    while t < duration:
        end = min(duration, t + chunk_sec)
        if end - t >= 5.0:
            segs.append({"start": round(t, 2), "end": round(end, 2)})
        t = end
    return segs


async def _audio_analysis_tags(audio_path: str) -> list[str]:
    """오디오 → 화자별 발화 감정 + 시간별 BGM 태그 리스트.

    파이프라인:
        1. /separate → vocals (보컬 분리)
        2. /diarize on vocals → 화자 segments
        3. /ser on vocals (화자 segments) → 화자별 감정
        4. /bgm_analyze on full audio (30초 청크) → 시간별 BGM
    """
    duration = await _audio_duration(audio_path)
    if duration <= 0:
        return []

    uid = _uuid.uuid4().hex
    vocals_path = f"/tmp/claude_aa_vocals_{uid}.wav"
    target = audio_path  # 보컬 분리 실패 시 원본 사용
    try:
        if await _aa_separate(audio_path, vocals_path):
            target = vocals_path

        # 화자 분리 + BGM 청크는 병렬
        diarize_task = asyncio.create_task(_analyze_audio_call("diarize", target))
        bgm_chunks = _build_chunks(duration)
        bgm_task = asyncio.create_task(_analyze_audio_call("bgm_analyze", audio_path, bgm_chunks))
        diarize_res, bgm_res = await asyncio.gather(diarize_task, bgm_task)

        speaker_segs = (diarize_res or {}).get("segments", []) if isinstance(diarize_res, dict) else []
        ser_segs = speaker_segs if speaker_segs else bgm_chunks  # 화자 분리 실패 시 시간 청크
        ser_res = await _analyze_audio_call("ser", target, ser_segs)

        # 화자 라벨링 (발화 시간 많은 순으로 화자1, 화자2, ...)
        speaker_labels: dict[str, str] = {}
        if speaker_segs:
            durations: dict[str, float] = {}
            for s in speaker_segs:
                durations[s["speaker"]] = durations.get(s["speaker"], 0.0) + (s["end"] - s["start"])
            for idx, (spk, _) in enumerate(sorted(durations.items(), key=lambda x: -x[1]), start=1):
                speaker_labels[spk] = f"화자{idx}"

        show_time = len(bgm_chunks) > 1 or bool(speaker_segs)

        def _t(start: float, end: float) -> str:
            return f"{int(start)}~{int(end)}s " if show_time else ""

        tags: list[str] = []
        if isinstance(ser_res, list):
            ser_by_start = {round(r.get("start", 0), 2): r for r in ser_res}
            if speaker_segs:
                for spk in speaker_segs:
                    ser = ser_by_start.get(round(spk["start"], 2))
                    if not ser:
                        continue
                    lbl = speaker_labels.get(spk["speaker"], "?")
                    tags.append(
                        f"[음성 {lbl} {_t(spk['start'], spk['end'])}{ser.get('label','?')}/{ser.get('intensity',0):.2f}]"
                    )
            else:
                for seg in ser_res:
                    tags.append(
                        f"[음성 {_t(seg.get('start',0), seg.get('end',0))}"
                        f"{seg.get('label','?')}/{seg.get('intensity',0):.2f}]"
                    )

        if isinstance(bgm_res, list):
            for seg in bgm_res:
                tags.append(
                    f"[BGM {_t(seg.get('start',0), seg.get('end',0))}"
                    f"{seg.get('tempo',0):.0f}BPM, {seg.get('key','?')}, {seg.get('mood','')}]"
                )
        return tags
    finally:
        if os.path.exists(vocals_path):
            try:
                os.unlink(vocals_path)
            except Exception:
                pass


def _strip_live2d_tags(text: str) -> str:
    """Live2D 표정/시선/볼륨 태그를 제거. tool-approval 팝업 intent 표시용으로 정리."""
    return _LIVE2D_TAG_RE.sub("", text).strip()


def _save_reasoning(text: str, session: Optional[str] = None) -> None:
    """last_reasoning에서 툴 승인 팝업용 intent를 추출해 세션별 파일에 저장.

    추출 우선순위:
    1. {의도} 패턴 — 마지막 중괄호 안 텍스트 (툴 승인 팝업 전용 포맷)
    2. [알림: ...] 패턴 — 마지막 알림 (fallback, {} 없을 때만)
    3. 둘 다 없으면 저장 안 함

    session: 호출자(LettaAgent)의 conversation_id. 넘기면 세션 바인딩이 없는
    호출에서도 UI 활성 세션 기준 경로가 되어 hook 쪽 경로와 어긋나지 않는다.
    """
    if not text:
        return
    cleaned = _strip_live2d_tags(text)
    if not cleaned:
        return

    # 1차: {의도} 패턴 추출 — 중괄호 안에 중괄호가 없는 단순 패턴
    intent_matches = list(re.finditer(r'\{([^{}]+)\}', cleaned, re.DOTALL))
    if intent_matches:
        intent = intent_matches[-1].group(1).strip()
    else:
        # 2차: [알림: ...] 패턴 fallback
        notify_matches = list(re.finditer(r'\[알림:\s*(.*?)\]', cleaned, re.DOTALL))
        intent = notify_matches[-1].group(1).strip() if notify_matches else ""

    if not intent:
        return
    session = session or _SESSION_CTX.get() or _DEFAULT_SESSION_KEY
    try:
        # 세션별 파일에 쓴다. hook 은 MIZUKI_INTENT_FILE env 로 같은 경로만 읽으므로
        # 동시에 두 세션이 돌아도 서로의 의도 문구가 섞이지 않는다.
        with open(reasoning_file(session), "w", encoding="utf-8") as _rf:
            # session: 어느 세션에서 쓴 것인지 확인용 꼬리표(디버깅/하위호환)
            json.dump(
                {
                    "text": intent,
                    "ts": _time.time(),
                    "session": session,
                },
                _rf,
                ensure_ascii=False,
            )
    except Exception:
        pass


async def _download_youtube_video(url: str, uid: str) -> Optional[str]:
    """yt-dlp_macos로 YouTube 영상을 /tmp에 다운로드. 실제 파일 경로 반환, 실패 시 None."""
    output_template = f"/tmp/claude_yt_{uid}.%(ext)s"
    try:
        proc = await asyncio.create_subprocess_exec(
            _YT_DLP_PATH, "-t", "mp4", "-o", output_template, url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_YT_DOWNLOAD_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning(f"[YouTube] 다운로드 타임아웃 ({_YT_DOWNLOAD_TIMEOUT}s): {url}")
            return None
        if proc.returncode != 0:
            logger.warning(f"[YouTube] 다운로드 실패 (exit {proc.returncode}): {stderr.decode()[:300]}")
            return None
        files = _glob.glob(f"/tmp/claude_yt_{uid}.*")
        if files:
            logger.info(f"[YouTube] 다운로드 완료: {files[0]}")
            return files[0]
        return None
    except Exception as e:
        logger.warning(f"[YouTube] 다운로드 오류: {e}")
        return None


async def _extract_frames(video_path: str, frame_paths: List[str]) -> List[str]:
    """ffmpeg으로 영상을 N등분해서 각 구간 중앙 프레임을 JPEG로 추출. 성공 경로 목록 반환.

    모든 프레임을 asyncio.gather로 병렬 추출해서 순차 대비 ~N배 빠름.
    """
    try:
        probe = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", video_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        probe_out, _ = await probe.communicate()
        duration = float(probe_out.decode().strip())

        n = len(frame_paths)

        async def _extract_one(i: int, frame_path: str) -> Optional[str]:
            ts = duration * (i + 0.5) / n
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
                "-frames:v", "1", "-q:v", "3", frame_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return frame_path if os.path.exists(frame_path) else None

        results = await asyncio.gather(
            *[_extract_one(i, p) for i, p in enumerate(frame_paths)],
            return_exceptions=True,
        )
        saved = [r for r in results if isinstance(r, str)]

        logger.info(f"[YouTube] {len(saved)}/{n}개 프레임 추출 완료 (병렬)")
        return saved
    except Exception as e:
        logger.warning(f"[YouTube] 프레임 추출 오류: {e}")
        return []


async def _attach_youtube_frames(text: str) -> str:
    """[유튜브 링크] 태그별로 영상을 다운로드하고 10개 프레임을 [스크린샷] 태그로 바로 뒤에 삽입."""
    matches = list(_YOUTUBE_TAG_RE.finditer(text))
    if not matches:
        return text

    for m in reversed(matches):
        url = m.group(1).strip()
        uid = _uuid.uuid4().hex
        frame_paths = [f"/tmp/claude_yt_frame_{uid}_{i:02d}.jpg" for i in range(_YT_FRAME_COUNT)]
        audio_path = f"/tmp/claude_yt_audio_{uid}.wav"
        video_path = None
        try:
            video_path = await _download_youtube_video(url, uid)
            if not video_path:
                continue
            saved_frames = await _extract_frames(video_path, frame_paths)
            if not saved_frames:
                continue
            frame_tags_list = [
                f"[스크린샷 source=youtube_frame]{p}[/스크린샷]" for p in saved_frames
            ]
            # 오디오 추출 + 분석 (실패해도 프레임만으로 진행)
            audio_tags_list: list[str] = []
            if await _extract_audio(video_path, audio_path):
                audio_tags_list = await _audio_analysis_tags(audio_path)
                if audio_tags_list:
                    logger.info(f"[YouTube] 오디오 분석 태그: {audio_tags_list}")

            insertion = m.group(0) + "\n" + "\n".join(frame_tags_list + audio_tags_list)
            text = text[: m.start()] + insertion + text[m.end():]
            logger.info(f"[YouTube] {len(saved_frames)}개 프레임 + 오디오 분석 삽입: {url}")
        except Exception as e:
            logger.warning(f"[YouTube] 미디어 처리 오류 ({url}): {e}")
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except Exception:
                    pass
            if os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    return text


# ---------------------------------------------------------------------------
# 일반 URL 분류 및 태그: [유튜브 링크] / [이미지 링크] / [웹 링크]
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

# YouTube 영상 URL 패턴 (watch?v=, youtu.be/, shorts/ 형식)
_YT_VIDEO_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/(?:watch(?:\?[^\s]*)?|shorts/)|youtu\.be/)[\w\-]+"
)
# YouTube 도메인 전체 (채널, 게시물 등 포함)
_YT_DOMAIN_RE = re.compile(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/")


_SOCIAL_OG_DOMAIN_RE = re.compile(
    r"https?://(?:www\.)?(?:x\.com|twitter\.com|instagram\.com)/"
)


def _head_content_type(url: str, timeout: int = 10) -> str:
    """HEAD 요청으로 Content-Type 반환. 실패 시 빈 문자열."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers=_YT_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.headers.get("Content-Type", "").lower()
    except Exception:
        return ""


def _fetch_og_image_url(url: str, timeout: int = 15) -> str:
    """소셜 미디어 URL에서 og:image URL 추출. 실패 시 빈 문자열."""
    try:
        req = urllib.request.Request(url, headers=_YT_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read(200 * 1024).decode("utf-8", errors="ignore")
        m = re.search(
            r'<meta[^>]+?(?:property|name)=["\']og:image["\'][^>]+?content=["\']([^"\']+)["\']',
            html,
        )
        if m:
            return m.group(1).strip()
    except Exception as e:
        logger.debug(f"[og:image] fetch 실패: {url} — {e}")
    return ""


def _fetch_image_base64(url: str, timeout: int = 15) -> tuple:
    """이미지 URL에서 (base64_str, mime_type) 반환. 실패 시 ('', '') 반환."""
    try:
        req = urllib.request.Request(url, headers=_YT_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            mime_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            return _base64.b64encode(data).decode(), mime_type
    except Exception as e:
        logger.warning(f"[image_display] 이미지 fetch 실패: {url} — {e}")
        return "", ""


async def _classify_url(url: str) -> str:
    """URL 콘텐츠 타입 분류: 'youtube_video' | 'image' | 'web'.

    YouTube 도메인은 HEAD 요청 없이 패턴으로 직접 분류해
    영상 URL과 채널/게시물/쇼츠를 정확히 구분한다.
    """
    if _YT_DOMAIN_RE.match(url):
        return "youtube_video" if _YT_VIDEO_RE.match(url) else "web"
    ct = await asyncio.to_thread(_head_content_type, url)
    if any(img_ct in ct for img_ct in _IMG_CONTENT_TYPES):
        return "image"
    path = url.lower().split("?")[0].split("#")[0]
    if any(path.endswith(ext) for ext in _IMG_EXTENSIONS):
        return "image"
    return "web"


async def _tag_general_urls(text: str) -> str:
    """[유튜브/이미지/웹 링크] 외의 URL을 적절한 태그로 감쌈.

    - YouTube 영상 URL → [유튜브 링크]
    - YouTube 채널/게시물 등 → [웹 링크]
    - 이미지 URL → [이미지 링크]
    - 그 외 → [웹 링크]
    """
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

    # 소셜 미디어 URL에서 og:image 추출 (x.com, twitter.com, instagram.com)
    social_urls = [url for url in unique_urls if _SOCIAL_OG_DOMAIN_RE.match(url)]
    og_images: dict[str, str] = {}
    if social_urls:
        og_results = await asyncio.gather(*[
            asyncio.to_thread(_fetch_og_image_url, url) for url in social_urls
        ])
        og_images = {url: img for url, img in zip(social_urls, og_results) if img}

    for url in to_process:
        url_type = type_map[url]
        if url_type == "youtube_video":
            text = text.replace(url, f"[유튜브 링크]{url}[/유튜브 링크]", 1)
        elif url_type == "image":
            text = text.replace(url, f"[이미지 링크]{url}[/이미지 링크]", 1)
        else:
            replacement = f"[웹 링크]{url}[/웹 링크]"
            # 소셜 미디어 og:image가 있으면 웹 링크 뒤에 이미지 링크도 추가
            if url in og_images:
                og_url = og_images[url]
                logger.info(f"[og:image] {url} → {og_url}")
                replacement += f" [이미지 링크]{og_url}[/이미지 링크]"
            text = text.replace(url, replacement, 1)

    return text


def _sync_system_prompt_if_changed(base_url: str, agent_id: str, expected_system: str) -> None:
    """conf.yaml에 letta_agent.id가 박혀있는 경우 대비, 서버 시작 시 한 번
    Letta 서버에 저장된 system_prompt와 코드에서 계산한 system_prompt를 비교해서
    다르면 자동으로 update. tool_prompts 파일 수정/추가/제거가 즉시 반영됨.

    실패해도 서버 기동 자체는 막지 않음 — log만 남김.
    """
    try:
        sync_client = Letta(base_url=base_url)
        agent = sync_client.agents.retrieve(agent_id)
        current = (agent.system or "")
        if current.strip() == (expected_system or "").strip():
            logger.info(f"[Letta sync] agent {agent_id}: system_prompt 변경 없음 (skip)")
            return
        logger.info(
            f"[Letta sync] agent {agent_id}: system_prompt 변경 감지 — 자동 update "
            f"(stored {len(current)}자 → new {len(expected_system)}자)"
        )
        sync_client.agents.update(agent_id, system=expected_system)
        logger.info(f"[Letta sync] agent {agent_id}: system_prompt update 완료")
    except Exception as e:
        logger.warning(f"[Letta sync] system_prompt 자동 동기화 실패 (무시): {e}")


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
    "archival_memory_delete",
]
_REQUIRED_BLOCKS = ["human", "persona"]
_MEMORY_WRITE_TOOLS = frozenset({
    "core_memory_append",
    "core_memory_replace",
    "archival_memory_insert",
    "archival_memory_delete",
})


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


# ── 재접속 대화 복원 (TODO19 1단계) ─────────────────────────────────────────
# 프론트는 접속할 때마다 빈 화면으로 시작한다. 백엔드의 chat_history JSON은
# 접속마다 새 history_uid가 생겨 조각나 있어 쓸모가 없으므로,
# 진짜 대화 원본인 letta conversation에서 직접 읽어 복원한다.
# letta 원문에는 프롬프트 메타 헤더와 Live2D 태그가 섞여 있어 그대로 보여주면
# 지저분하다 → 아래 정제 함수로 사람이 읽는 부분만 남긴다.
_HISTORY_META_LINE_RE = re.compile(
    r"^\[(?:현재 TTS 볼륨|현재 시각|PROACTIVE_USER_INSTRUCTION)[^\]]*\]\s*$",
    re.MULTILINE,
)
_NEMULIN_BLOCK_RE = re.compile(r"\[네무린\](.*?)\[/네무린\]", re.DOTALL)
_HISTORY_SKIP_PAYLOAD_TYPES = {"system_alert", "heartbeat", "login", "logout"}


def _clean_user_history_text(raw: Optional[str]) -> Optional[str]:
    """letta user_message → 프론트 채팅창에 보여줄 네무린 발화. 숨길 것이면 None."""
    text = (raw or "").strip()
    if not text:
        return None

    # 컴팩션 안내 같은 시스템 JSON 페이로드는 대화가 아니므로 숨긴다.
    if text.startswith("{"):
        payload = None
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            if payload.get("type") in _HISTORY_SKIP_PAYLOAD_TYPES:
                return None
            inner = payload.get("message")
            if isinstance(inner, str):
                text = inner.strip()

    # 자발 발화 트리거는 네무린이 친 말이 아니다.
    if "[PROACTIVE_SPEAK_TRIGGER]" in text:
        return None

    blocks = [b.strip() for b in _NEMULIN_BLOCK_RE.findall(text) if b.strip()]
    if blocks:
        text = "\n".join(blocks)
    else:
        text = _HISTORY_META_LINE_RE.sub("", text)

    return text.strip() or None


def _clean_assistant_history_text(raw: Optional[str]) -> Optional[str]:
    """letta assistant_message → 프론트 채팅창 표시용 텍스트."""
    text = raw or ""
    text = re.sub(r"\[알림:\s*.*?\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[이미지\].*?\[/이미지\]", "", text, flags=re.DOTALL)
    text = _MIZUKI_IMAGE_TAG_RE.sub("", text)
    text = _SESSION_GIST_TAG_RE.sub("", text)
    # 닫는 태그가 없는 경우에도 요약문이 채팅창에 남지 않게 줄 끝까지 제거
    text = _SESSION_GIST_OPEN_ONLY_RE.sub("", text)
    text = _LIVE2D_TAG_RE.sub("", text)
    text = text.replace("[무음시작]", "").replace("[무음끝]", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or None


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
        conversation_id: str = _DEFAULT_SESSION_KEY,
    ):
        super().__init__()
        self.url = f"http://{host}:{port}"

        if not id:
            id = _select_or_create_agent(self.url, system_prompt, model, embedding, context_window_limit)
        else:
            # conf.yaml에 id가 박혀있어 _select_or_create_agent를 거치지 않는 경우에도,
            # 서버 시작 시 한 번 system_prompt diff를 체크해서 변경되었으면 자동 동기화.
            _sync_system_prompt_if_changed(self.url, id, system_prompt)

        self.client = AsyncLetta(base_url=self.url, timeout=7200)
        self.id = id
        # 다중 세션: letta conversation 하나 = 프론트 세션 하나.
        # "default"는 기존 단일 세션(messages.conversation_id IS NULL)을 가리키는 특수 키로
        # letta 서버가 agent-direct 모드로 처리한다 → 기존 대화 내역이 그대로 이어진다.
        # UI 활성 세션(= 세션 바인딩이 없는 호출의 기본값).
        self._active_conversation_id: str = conversation_id or _DEFAULT_SESSION_KEY
        # 세션별 상태 저장소. 동시에 여러 세션이 돌아도 서로 간섭하지 않는다.
        # 키는 conversation_id, 값은 사용자 지정 이름 / 인터럽트 플래그.
        self._session_names: Dict[str, str] = {
            _DEFAULT_SESSION_KEY: _DEFAULT_SESSION_NAME
        }
        if self._active_conversation_id != _DEFAULT_SESSION_KEY:
            self._session_names.setdefault(self._active_conversation_id, "")
        self._abort_flags: Dict[str, bool] = {}
        # ADB 스크린샷 실시간 전송용 WebSocket send 함수는 _SEND_FN_CTX(ContextVar)에
        # 보관한다 → property _websocket_send_fn 참고. 세션마다 다른 값을 쓰기 때문.
        logger.info(
            f"LettaAgent 초기화 완료. agent_id={self.id}, url={self.url}, "
            f"conversation_id={self.conversation_id}"
        )

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

    # ── 세션 상태 (ContextVar 바인딩 우선, 없으면 UI 활성 세션) ────────────
    # 모든 세션 의존 값을 property 로 계산한다. capture task 가 bind_session()
    # 으로 자기 세션을 고정했으면 그 값, 아니면 UI 활성 세션을 쓴다.

    @property
    def conversation_id(self) -> str:
        return _SESSION_CTX.get() or self._active_conversation_id

    @conversation_id.setter
    def conversation_id(self, value: Optional[str]) -> None:
        self._active_conversation_id = value or _DEFAULT_SESSION_KEY

    @property
    def active_conversation_id(self) -> str:
        """세션 바인딩을 무시한 UI 활성 세션."""
        return self._active_conversation_id

    @property
    def _session_key(self) -> str:
        return _sanitize_session_key(self.conversation_id)

    @property
    def _notify_file(self) -> str:
        return progress_notify_file(self._session_key)

    @property
    def _abort_file(self) -> str:
        return abort_flag_file(self._session_key)

    @property
    def _session_name(self) -> str:
        cid = self.conversation_id
        default = _DEFAULT_SESSION_NAME if cid == _DEFAULT_SESSION_KEY else ""
        return self._session_names.get(cid, default)

    @_session_name.setter
    def _session_name(self, value: Optional[str]) -> None:
        self._session_names[self.conversation_id] = (value or "").strip()

    @property
    def _websocket_send_fn(self) -> Optional[Any]:
        return _SEND_FN_CTX.get()

    @_websocket_send_fn.setter
    def _websocket_send_fn(self, value: Optional[Any]) -> None:
        _SEND_FN_CTX.set(value)

    @property
    def _abort_flag(self) -> bool:
        return self._abort_flags.get(self.conversation_id, False)

    @_abort_flag.setter
    def _abort_flag(self, value: bool) -> None:
        self._abort_flags[self.conversation_id] = bool(value)

    async def fetch_history_messages(
        self,
        limit: int = 50,
        before: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """letta conversation에서 과거 대화를 읽어 프론트 복원용 목록으로 변환.

        Args:
            limit: 돌려줄 표시용 메시지 개수 상한
            before: 이 message id보다 과거만 조회 (위로 스크롤 추가 로드용)
            conversation_id: 조회 대상 세션. 생략하면 이 에이전트의 현재 세션.

        Returns:
            {"messages": [{"id", "role", "content", "timestamp"}],
             "oldest_id": str|None, "has_more": bool}
        """
        want = max(1, min(int(limit or 50), 200))
        # 메타 메시지가 정제 과정에서 걸러지므로 넉넉히 받아온다.
        fetch_n = min(want * 3, 400)
        target = conversation_id or self.conversation_id

        kwargs: Dict[str, Any] = {
            "agent_id": self.id,
            "limit": fetch_n,
            "order": "desc",
            "extra_query": {
                "include_return_message_types": [
                    "user_message",
                    "assistant_message",
                ]
            },
        }
        if before:
            kwargs["before"] = before

        try:
            page = await self.client.conversations.messages.list(target, **kwargs)
        except Exception as e:
            logger.error(f"[history] letta 대화 복원 실패 (conv={target}): {e}")
            return {"messages": [], "oldest_id": None, "has_more": False}

        raw_items = list(getattr(page, "items", None) or [])
        # 서버가 order와 무관하게 정렬해 돌려주는 사례가 있어 직접 오름차순 정렬.
        raw_items.sort(key=lambda m: str(getattr(m, "date", "") or ""))

        cleaned: List[Dict[str, Any]] = []
        for msg in raw_items:
            mtype = getattr(msg, "message_type", "")
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                content = "" if content is None else str(content)

            if mtype == "user_message":
                text = _clean_user_history_text(content)
                role = "human"
            elif mtype == "assistant_message":
                text = _clean_assistant_history_text(content)
                role = "ai"
            else:
                continue

            if not text:
                continue

            date = getattr(msg, "date", None)
            cleaned.append(
                {
                    "id": str(getattr(msg, "id", "") or ""),
                    "role": role,
                    # 프론트는 type 으로 표시 모양을 고른다 (text / progress_update).
                    "type": "text",
                    "content": text,
                    "timestamp": str(date) if date else "",
                }
            )

        has_more = len(raw_items) >= fetch_n
        if len(cleaned) > want:
            cleaned = cleaned[-want:]
            has_more = True

        oldest_id = cleaned[0]["id"] if cleaned else None
        logger.info(
            f"[history] 복원 {len(cleaned)}건 (conv={target}, raw={len(raw_items)}, "
            f"before={before}, has_more={has_more})"
        )
        return {
            "messages": cleaned,
            "oldest_id": oldest_id,
            "has_more": has_more,
        }

    # ------------------------------------------------------------------
    # 다중 세션(letta conversation) 관리 — TODO19 2단계
    #   세션 하나 = letta conversation 하나.
    #   core memory(human/persona)와 archival은 agent 단위라 모든 세션이 공유한다.
    #   분리되는 것은 "진행 중인 대화 흐름"뿐 (TODO19 필수 제약).
    # ------------------------------------------------------------------

    @staticmethod
    def _conv_to_dict(conv: Any) -> Dict[str, Any]:
        """letta Conversation 객체를 프론트 전송용 dict로."""
        summary = getattr(conv, "summary", None) or ""
        name, gist = _decode_session_meta(summary)
        return {
            "id": str(getattr(conv, "id", "") or ""),
            # summary 는 하위호환용(이름). 프론트는 name/gist 를 쓴다.
            "summary": name,
            "name": name,
            "gist": gist,
            "created_at": str(getattr(conv, "created_at", "") or ""),
            "last_message_at": str(getattr(conv, "last_message_at", "") or ""),
            "is_default": False,
        }

    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """세션 목록. 맨 앞은 항상 기존 단일 대화(default) 가상 세션.

        letta의 conversation 목록 API는 default(conversation_id IS NULL)를
        돌려주지 않으므로 프론트에서 쓸 수 있게 직접 끼워 넣는다.
        """
        _dname, _dgist = _load_default_meta()
        sessions: List[Dict[str, Any]] = [
            {
                "id": _DEFAULT_SESSION_KEY,
                "summary": _dname,
                "name": _dname,
                "gist": _dgist,
                "created_at": "",
                "last_message_at": "",
                "is_default": True,
            }
        ]
        try:
            page = await self.client.conversations.list(
                agent_id=self.id,
                limit=max(1, min(int(limit or 100), 200)),
            )
        except Exception as e:
            logger.error(f"[session] conversation 목록 조회 실패: {e}")
            return sessions

        # conversations.list 는 response_model=List[Conversation] 이라
        # 페이지 객체가 아니라 그냥 리스트를 돌려준다 (2026-10-24 실측).
        items = list(getattr(page, "items", None) or page or [])
        converted = [self._conv_to_dict(c) for c in items]
        # 최근 발화 순. last_message_at이 비어 있으면 created_at으로 대체.
        converted.sort(
            key=lambda s: s["last_message_at"] or s["created_at"],
            reverse=True,
        )
        sessions.extend(converted)
        logger.info(f"[session] 목록 {len(sessions)}건 (default 포함)")
        return sessions

    async def create_session(self, summary: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """새 세션(conversation) 생성. 기억 블록은 agent 공유이므로 isolated 지정 안 함.

        summary 인자는 '사용자가 정한 세션 이름'이다. 한 줄 요약은 대화가 시작된 뒤
        미즈키가 [세션요약] 태그로 채운다.
        """
        try:
            conv = await self.client.conversations.create(
                agent_id=self.id,
                summary=_encode_session_meta(summary or "", ""),
            )
        except Exception as e:
            logger.error(f"[session] conversation 생성 실패: {e}")
            return None
        result = self._conv_to_dict(conv)
        logger.info(f"[session] 생성 완료 id={result['id']}")
        return result

    async def delete_session(self, conversation_id: str) -> bool:
        """세션 삭제. default는 삭제 불가."""
        if not conversation_id or conversation_id == _DEFAULT_SESSION_KEY:
            logger.warning("[session] default 세션은 삭제할 수 없음")
            return False
        try:
            await self.client.conversations.delete(conversation_id)
        except Exception as e:
            logger.error(f"[session] 삭제 실패 (id={conversation_id}): {e}")
            return False
        logger.info(f"[session] 삭제 완료 id={conversation_id}")
        if conversation_id == self.conversation_id:
            self.set_conversation(_DEFAULT_SESSION_KEY)
        return True

    async def _fetch_session_meta(self, conversation_id: str) -> tuple:
        """conversation 하나의 (이름, 요약) 조회. 실패하면 ("", "")."""
        if conversation_id == _DEFAULT_SESSION_KEY:
            return _load_default_meta()
        try:
            conv = await self.client.conversations.retrieve(conversation_id)
        except Exception as e:
            logger.debug(f"[session] 메타 조회 실패 (id={conversation_id}): {e}")
            return "", ""
        return _decode_session_meta(getattr(conv, "summary", None))

    async def resolve_session_by_key(self, session_key: str) -> tuple:
        """세션 파일 키 → (conversation_id, 표시용 세션 이름).

        `claude -p` 의 PreToolUse hook 은 자기가 속한 세션을 파일명 안전 문자로
        치환된 키(`_sanitize_session_key` 결과)로만 알고 있다. 승인 팝업에
        "어느 방의 요청인지" 적어주려면 원본 conversation_id 와 사용자가 정한
        이름이 필요하므로, 알고 있는 세션들을 같은 규칙으로 변환해 역매핑한다.
        캐시에 이름이 없으면 letta 에서 한 번 조회해 보강한다.
        """
        key = _sanitize_session_key(session_key)
        target = ""
        for cid in list(self._session_names.keys()):
            if _sanitize_session_key(cid) == key:
                target = cid
                break
        if not target:
            # 아직 전환/생성한 적 없는 세션. default 는 키가 그대로 id 다.
            target = _DEFAULT_SESSION_KEY if key == _DEFAULT_SESSION_KEY else key

        name = (self._session_names.get(target) or "").strip()
        if not name:
            fetched, _ = await self._fetch_session_meta(target)
            name = (fetched or "").strip()
            if name:
                self._session_names[target] = name
        if not name and target == _DEFAULT_SESSION_KEY:
            name = _DEFAULT_SESSION_NAME
        return target, name

    async def update_session_summary(self, conversation_id: str, summary: str) -> bool:
        """세션 이름 변경. 미즈키가 쓴 한 줄 요약은 그대로 둔다."""
        name = (summary or "").strip()
        if not conversation_id:
            return False
        if conversation_id == _DEFAULT_SESSION_KEY:
            _, gist = _load_default_meta()
            _save_default_meta(name or _DEFAULT_SESSION_NAME, gist)
            self._session_names[_DEFAULT_SESSION_KEY] = name or _DEFAULT_SESSION_NAME
            return True
        _, gist = await self._fetch_session_meta(conversation_id)
        try:
            await self.client.conversations.update(
                conversation_id, summary=_encode_session_meta(name, gist)
            )
        except Exception as e:
            logger.error(f"[session] 이름 수정 실패 (id={conversation_id}): {e}")
            return False
        self._session_names[conversation_id] = name
        return True

    async def update_session_gist(self, conversation_id: str, gist: str) -> bool:
        """미즈키가 남긴 한 줄 요약 저장. 사용자가 정한 이름은 건드리지 않는다."""
        text = (gist or "").strip()
        if not conversation_id or not text:
            return False
        if conversation_id == _DEFAULT_SESSION_KEY:
            name, _ = _load_default_meta()
            _save_default_meta(name, text)
            logger.info(f"[session] 요약 갱신 (default): {text}")
            return True
        name, _ = await self._fetch_session_meta(conversation_id)
        try:
            await self.client.conversations.update(
                conversation_id, summary=_encode_session_meta(name, text)
            )
        except Exception as e:
            logger.error(f"[session] 요약 갱신 실패 (id={conversation_id}): {e}")
            return False
        logger.info(f"[session] 요약 갱신 (id={conversation_id}): {text}")
        return True

    def set_conversation(
        self, conversation_id: Optional[str], name: Optional[str] = None
    ) -> str:
        """활성 세션 전환.

        conversation_id에 따라 세션별 IPC 파일 경로(중간 알림/인터럽트 플래그)도 같이 갈아탄다.
        letta 쪽 claude_cli_client.py가 CONVERSATION_ID 메타데이터로 같은 경로를 계산하므로
        두 프로세스의 세션 키가 어긋나지 않아야 한다.
        """
        target = conversation_id or _DEFAULT_SESSION_KEY
        if name is not None:
            self._session_names[target] = (name or "").strip()
        elif target not in self._session_names:
            # 이름을 못 받았으면 다음 발화 때 다시 채우도록 비워둔다.
            self._session_names[target] = (
                _DEFAULT_SESSION_NAME if target == _DEFAULT_SESSION_KEY else ""
            )
        if target == self._active_conversation_id:
            return target
        # 진행 중인 다른 세션의 abort 플래그는 건드리지 않는다(동시 실행 보호).
        self._active_conversation_id = target
        self._abort_flags[target] = False
        logger.info(
            f"[session] 활성 세션 전환 → {target} "
            f"(key={_sanitize_session_key(target)}, "
            f"notify={progress_notify_file(_sanitize_session_key(target))})"
        )
        return target

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        # The Letta Server automatically stores historical messages, so this part is not needed
        pass

    def handle_interrupt(
        self, heard_response: str, conversation_id: Optional[str] = None
    ) -> None:
        """인터럽트 신호 수신 시 _chat_raw 스트림 루프와 letta 서버의 claude -p subprocess를 종료.

        1. _abort_flag = True → 스트리밍 루프 조기 break
        2. abort 파일 생성 → letta 서버의 claude_cli_client.py watcher가 감지해서 proc.kill()
        """
        target = conversation_id or self.conversation_id
        self._abort_flags[target] = True
        abort_path = abort_flag_file(_sanitize_session_key(target))
        try:
            with open(abort_path, "w", encoding="utf-8") as _af:
                _af.write(str(_time.time()))
            logger.info(f"[LettaAgent] abort 파일 생성: {abort_path}")
        except Exception as _e:
            logger.warning(f"[LettaAgent] abort 파일 생성 실패 (무시): {_e}")
        logger.info(
            f"[LettaAgent] 인터럽트 신호 수신 (session={target}) "
            "→ _abort_flag + abort 파일 설정"
        )

    async def chat(  # type: ignore[override]
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        async for item in self._chat(input_data):
            yield item

    async def _chat_raw(self, input_data: BatchInput) -> AsyncIterator[str]:
        """Raw token generator. Decorated into self._chat during __init__."""
        # 새 대화 시작 시 abort 플래그 및 파일 초기화
        self._abort_flag = False
        try:
            if os.path.exists(self._abort_file):
                os.unlink(self._abort_file)
        except Exception:
            pass

        # 세션 활동 상태 표시: 프론트 채팅창 마지막에 스피너를 띄우기 위한 이벤트.
        # 여기서는 "active + detail" 로 시작 알림만 보내고, finally 블록에서 idle 이벤트를 웹소켓으로 직접 전송.
        yield {"type": "agent_status", "status": "active", "detail": "생각 중", "timestamp": _time.time()}
        input_content: Union[str, List[InputUnionMember1]]
        text = self._to_text_prompt(input_data)
        text = await _tag_general_urls(text)
        text = await _preprocess_youtube_links(text)
        text = await _attach_youtube_frames(text)

        # ── 이미지 URL 감지: [이미지 링크]URL[/이미지 링크] 패턴 추출 후 프론트 전송 ──
        _IMAGE_LINK_RE = re.compile(r'\[이미지 링크\](https?://[^\[\]]+?)\[/이미지 링크\]')
        for _m in _IMAGE_LINK_RE.finditer(text):
            _img_url = _m.group(1).strip()
            _img_b64, _mime_type = await asyncio.to_thread(_fetch_image_base64, _img_url)
            if _img_b64:
                logger.info(f"[image_display] URL 이미지 전송: {_img_url}")
                yield {"type": "image_display", "image_base64": _img_b64, "mime_type": _mime_type, "label": _img_url}

        if input_data.images:
            screenshot_tags: List[str] = []
            for img in input_data.images:
                img_data = img.data
                if img_data.startswith("data:") and "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                source = img.source.value if hasattr(img.source, "value") else str(img.source)
                mime = getattr(img, "media_type", None) or getattr(img, "mime_type", None) or "image/jpeg"
                # Read 도구는 확장자로 이미지 포맷을 판별하므로 mime 에 맞춰 붙인다.
                # (첨부 업로드는 PNG·WebP 등 JPEG 가 아닌 원본이 그대로 올라올 수 있다)
                ext = _IMAGE_EXT_BY_MIME.get(str(mime).strip().lower(), "jpg")
                temp_path = f"/tmp/claude_screenshot_{_uuid.uuid4().hex}.{ext}"
                await asyncio.to_thread(
                    lambda p=temp_path, d=img_data: open(p, "wb").write(_base64.b64decode(d))
                )
                # ago_seconds가 있으면 태그 속성으로 추가. 0이면 "현재", 양수면 "N초 전".
                ago = getattr(img, "ago_seconds", None)
                if ago is not None:
                    ago_attr = f" ago={ago:g}s"
                else:
                    ago_attr = ""
                # 직접 첨부한 파일은 원본 이름에 단서가 있는 경우가 많아 같이 넘긴다.
                raw_name = (getattr(img, "name", None) or "").strip()
                safe_name = raw_name.replace("[", "(").replace("]", ")").replace('"', "'")
                name_attr = f' name="{safe_name}"' if safe_name else ""
                screenshot_tags.append(
                    f"[스크린샷 source={source}{ago_attr}{name_attr}]{temp_path}[/스크린샷]"
                )
                logger.info(
                    f"[Screenshot] temp 저장: {temp_path} (source={source}{ago_attr}{name_attr})"
                )
                # ── 스크린샷도 프론트에 전송 ──
                yield {"type": "image_display", "image_base64": img_data, "mime_type": mime, "label": source}
            text = "\n".join(screenshot_tags) + "\n" + text

        # 현재 세션 이름을 발화 맨 앞에 붙여서, 미즈키가 지금 어느 세션에 있는지 알게 한다.
        # 네무린은 id가 아니라 이름으로 세션을 구분하므로 다른 세션 얘기를 할 때 필요하다.
        _sname = (self._session_name or "").strip()
        if not _sname:
            _sname, _ = await self._fetch_session_meta(self.conversation_id)
            self._session_name = _sname
        if _sname:
            text = f"[세션] {_sname} [/세션]\n" + text
        input_content = text

        # agents.messages.create는 letta에서 deprecated → conversations 엔드포인트 사용.
        # conversation_id="default" + agent_id는 letta의 agent-direct 모드로,
        # 기존 단일 세션 대화 내역(conversation_id IS NULL)을 그대로 이어받는다.
        stream = await self.client.conversations.messages.create(
            self.conversation_id,
            agent_id=self.id,
            streaming=True,  # type: ignore[call-overload]
            stream_tokens=True,
            background=False,  # 단일 워커 환경 — False로 스트리밍 타이밍 개선
            input=input_content,
            max_steps=50,
        )

        _yielded = 0
        last_reasoning = ""       # 직전 reasoning_message 누적 버퍼
        _assistant_buffer = ""    # assistant_message 누적 버퍼 (토큰 경계 넘어 [알림:] 감지용)
        # 발화 방식 분기 (머신별 설정)
        #   false(기본, 집): 여러 step 에 걸친 모든 assistant_message 를 그대로 흘려보낸다.
        #                    → 상위에서 captured 에 쌓였다가 전부 TTS 로 읽힌다.
        #   true(회사/codex): 툴을 쓰는 step 의 대사는 progress_update(텍스트)로만 내보내고
        #                    스트림 마지막 덩어리만 음성으로 읽는다.
        # 환경변수 MIZUKI_SPEAK_ONLY_FINAL=1 로 켠다.
        _speak_only_final = str(
            os.environ.get("MIZUKI_SPEAK_ONLY_FINAL", "")
        ).strip().lower() in ("1", "true", "yes", "on")
        _pending_speech = ""
        _deferred_final_speech = ""
        _had_tool_call = False
        _stream_failed = False
        _session_gist = ""        # [세션요약] 태그로 받은 현재 세션 한 줄 요약 (마지막 것 사용)
        _gist_open = False        # [세션요약] 가 열린 채 토큰 경계를 넘어갔는지
        _gist_carry = ""          # 청크 경계에서 쪼개진 세션요약 태그 조각 보류분
        _found_notifies: set = set()  # 이미 yield한 [알림:] 내용 — 중복 방지
        _found_images: set = set()    # 이미 yield한 [이미지] 파일 경로 — 중복 방지

        # -----------------------------------------------------------------------
        # 중간 진행 알림 실시간 표시:
        # claude_cli_client.py가 --output-format stream-json으로 실행하면서
        # [알림: ...] 감지 즉시 세션별 알림 파일(self._notify_file)에 JSONL로 기록한다.
        # 이 polling task가 0.3초마다 파일을 읽어 _notify_queue에 넣고,
        # Letta 스트림 루프에서 pending 알림을 먼저 yield한다.
        # -----------------------------------------------------------------------
        _notify_queue: asyncio.Queue = asyncio.Queue()
        _image_queue: asyncio.Queue = asyncio.Queue()  # ADB 스크린샷 전송용
        _polling_stop = asyncio.Event()
        _notify_file_offset: int = 0
        # 스트림 시작 시점의 /tmp/*.png mtime 전부 초기화 → 이전 스크린샷 오발송 방지
        _adb_screenshot_mtimes: dict = {
            _p: os.path.getmtime(_p)
            for _p in _glob.glob(os.path.join(_ADB_SCREENSHOT_DIR, "*.png"))
            if os.path.isfile(_p)
        }

        async def _poll_progress_file() -> None:
            """백그라운드: 알림 파일을 0.3초마다 polling."""
            nonlocal _notify_file_offset
            while not _polling_stop.is_set():
                try:
                    with open(self._notify_file, "r", encoding="utf-8") as _pf:
                        _pf.seek(_notify_file_offset)
                        for _line in _pf:
                            _line = _line.strip()
                            if not _line:
                                continue
                            try:
                                _item = json.loads(_line)
                                _txt = _item.get("text", "")
                                if _txt and _txt not in _found_notifies:
                                    await _notify_queue.put(_txt)
                            except Exception:
                                pass
                        _notify_file_offset = _pf.tell()
                except FileNotFoundError:
                    pass
                except Exception as _pe:
                    logger.debug(f"[Letta] 알림 파일 polling 오류 (무시): {_pe}")
                await asyncio.sleep(0.3)

        async def _poll_adb_screenshot() -> None:
            """백그라운드: /tmp/*.png 전체 감시 → 변경/신규 시 WebSocket 직접 전송 또는 image_queue에 추가.

            self._websocket_send_fn이 주입되어 있으면 generator yield 우회하여 즉시 전송.
            그렇지 않으면 기존 방식대로 _image_queue에 넣어서 flush 시 yield.
            """
            nonlocal _adb_screenshot_mtimes
            while not _polling_stop.is_set():
                try:
                    for _path in _glob.glob(os.path.join(_ADB_SCREENSHOT_DIR, "*.png")):
                        try:
                            _mtime = os.path.getmtime(_path)
                            _prev = _adb_screenshot_mtimes.get(_path, 0.0)
                            if _mtime > _prev:
                                _adb_screenshot_mtimes[_path] = _mtime
                                _img_bytes = await asyncio.to_thread(
                                    lambda p=_path: open(p, "rb").read()
                                )
                                _img_b64 = _base64.b64encode(_img_bytes).decode()
                                _label = os.path.basename(_path)
                                _img_payload = {
                                    "type": "image_display",
                                    "image_base64": _img_b64,
                                    "mime_type": "image/png",
                                    "label": _label,
                                }
                                if self._websocket_send_fn is not None:
                                    # WebSocket 직접 전송: await로 즉시 전송 (create_task는 스트림이 끝날 때까지 밀려 한꺼번에 실행됨)
                                    try:
                                        await self._websocket_send_fn(json.dumps(_img_payload))
                                        logger.info(f"[Letta] ADB 스크린샷 감지 ({_label}) → WebSocket 직접 전송")
                                    except Exception as _send_err:
                                        logger.debug(f"[Letta] WebSocket 직접 전송 실패 ({_label}), image_queue fallback: {_send_err}")
                                        await _image_queue.put({"image_base64": _img_b64, "label": _label})
                                else:
                                    await _image_queue.put({"image_base64": _img_b64, "label": _label})
                                    logger.info(f"[Letta] ADB 스크린샷 감지 ({_label}) → image_queue 추가 (fallback)")
                        except (FileNotFoundError, OSError):
                            pass
                except Exception as _ae:
                    logger.debug(f"[Letta] ADB 스크린샷 polling 오류 (무시): {_ae}")
                await asyncio.sleep(0.5)

        _poll_task = asyncio.create_task(_poll_progress_file())
        _adb_poll_task = asyncio.create_task(_poll_adb_screenshot())

        # 스트림 → 내부 큐 피딩: 도구 실행 중에도 이미지/알림 실시간 flush 가능하게 함
        _stream_queue: asyncio.Queue = asyncio.Queue()

        async def _feed_stream() -> None:
            nonlocal _stream_failed
            try:
                async for _t in stream:
                    await _stream_queue.put(_t)
            except Exception as _fe:
                _stream_failed = True
                logger.error(f"[Letta] _feed_stream 오류: {_fe}")
            finally:
                await _stream_queue.put(None)  # 종료 sentinel

        _stream_feed_task = asyncio.create_task(_feed_stream())

        async def _drain_notify_queue() -> None:
            """pending 중간 알림을 모두 yield (호출자 generator에서 inline으로 처리)."""
            # 참고: generator 내부에서 다른 generator를 yield from 할 수 없으므로
            # 이 함수는 직접 호출하지 않고, 루프 안에서 while+get_nowait로 처리한다.
            pass

        try:
            while True:
                # 인터럽트 신호 체크 — abort_flag가 설정되면 스트리밍 즉시 중단
                if self._abort_flag:
                    logger.info("[LettaAgent] _abort_flag 감지 → 스트리밍 루프 종료")
                    break

                # Letta 스트림 메시지 처리 전 pending 알림 먼저 flush
                while not _notify_queue.empty():
                    _pn = _notify_queue.get_nowait()
                    if _pn not in _found_notifies:
                        yield {"type": "progress_update", "text": _pn, "timestamp": _time.time()}
                        _found_notifies.add(_pn)

                # ADB 스크린샷 이미지 flush
                while not _image_queue.empty():
                    _pi = _image_queue.get_nowait()
                    yield {"type": "image_display", "image_base64": _pi["image_base64"], "mime_type": "image/png", "label": _pi["label"]}

                # 스트림 큐에서 다음 토큰 대기 (0.1초 timeout → 타임아웃 시 위 큐들 재flush)
                try:
                    token = await asyncio.wait_for(_stream_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if token is None:  # 스트림 종료 sentinel
                    break

                logger.debug(f"[Letta stream] message_type={token.message_type!r}")
                if token.message_type == "reasoning_message":
                    _rc = getattr(token, "content", "")
                    if isinstance(_rc, list):
                        _new_r = "".join(getattr(p, "text", "") for p in _rc)
                    else:
                        _new_r = _rc or ""
                    last_reasoning += _new_r  # 누적 (덮어쓰기 아님)
                    # [알림: 내용] 패턴 감지 → TTS 없이 채팅창에만 표시 (중복 방지)
                    for _m in re.finditer(r'\[알림:\s*(.*?)\]', last_reasoning, re.DOTALL):
                        _notify = _m.group(1).strip()
                        if _notify and re.sub(r'[.\s…]+', '', _notify) and _notify not in _found_notifies:
                            yield {"type": "progress_update", "text": _notify, "timestamp": _time.time()}
                            _found_notifies.add(_notify)
                    # reasoning을 파일에 즉시 저장 — hook이 다음 tool_call 전에 읽어갈 수 있도록
                    _save_reasoning(last_reasoning, self.conversation_id)
                    continue
                elif token.message_type == "assistant_message":
                    content = token.content
                    if isinstance(content, list):
                        text = "".join(part.text for part in content)
                    else:
                        text = content
                    # 누적 버퍼에 추가 — 토큰 경계를 넘어 [알림: ...] 감지
                    _assistant_buffer += text
                    # 이전 청크 끝에서 보류한 태그 조각을 복원 (버퍼에는 이미 반영돼 있음)
                    if _gist_carry:
                        text = _gist_carry + text
                        _gist_carry = ""
                    _notify_texts: list[str] = []
                    for _m in re.finditer(r'\[알림:\s*(.*?)\]', _assistant_buffer, re.DOTALL):
                        _notify = _m.group(1).strip()
                        if _notify and re.sub(r'[.\s…]+', '', _notify) and _notify not in _found_notifies:
                            yield {"type": "progress_update", "text": _notify, "timestamp": _time.time()}
                            _notify_texts.append(_notify)
                            _found_notifies.add(_notify)
                    # [이미지]path[/이미지] 또는 <image_display path="..." /> 태그 감지 → 파일 base64로 인코딩해 image_display 전송
                    for _im in _MIZUKI_IMAGE_TAG_RE.finditer(_assistant_buffer):
                        _img_path = (_im.group(1) or _im.group(2) or "").strip()
                        if not _img_path or _img_path in _found_images:
                            continue
                        _found_images.add(_img_path)  # 실패해도 재시도 방지
                        _img_b64, _img_mime, _img_label = await asyncio.to_thread(
                            _load_local_image_as_b64, _img_path
                        )
                        if _img_b64:
                            logger.info(f"[image_display] 미즈키 첨부 이미지 전송: {_img_path}")
                            _img_payload = {
                                "type": "image_display",
                                "image_base64": _img_b64,
                                "mime_type": _img_mime,
                                "label": _img_label,
                            }
                            # WebSocket 직접 전송이 가능하면 즉시 밀어 넣어서 순서 보장
                            if self._websocket_send_fn is not None:
                                try:
                                    await self._websocket_send_fn(json.dumps(_img_payload))
                                except Exception as _send_err:
                                    logger.debug(f"[image_display] WS 직접 전송 실패, yield로 fallback: {_send_err}")
                                    yield _img_payload
                            else:
                                yield _img_payload
                    # [세션요약]...[/세션요약] 감지 → 현재 세션 한 줄 요약으로 보관(마지막 것 채택)
                    # 닫는 태그를 빼먹어도 줄 끝까지를 요약으로 받아들인다
                    _session_gist = _extract_session_gist(_assistant_buffer, _session_gist)
                    # [알림: ...] 및 [이미지]path[/이미지] 태그를 TTS/표시 텍스트에서 제거
                    text = re.sub(r'\[알림:\s*.*?\]', '', text, flags=re.DOTALL)
                    text = _MIZUKI_IMAGE_TAG_RE.sub('', text)
                    # 세션요약 태그는 토큰 경계를 넘어가도 통째로 음성/표시에서 빠지게 상태로 처리
                    if _gist_open:
                        _close = text.find(_SESSION_GIST_CLOSE)
                        if _close == -1:
                            # 닫는 태그가 아직 안 왔다 — 꼬리에 쪼개진 조각이 있으면 보류
                            _n = _gist_tag_partial_suffix_len(text)
                            _gist_carry = text[-_n:] if _n else ""
                            text = ''
                        else:
                            text = text[_close + len(_SESSION_GIST_CLOSE):]
                            _gist_open = False
                    text = _SESSION_GIST_TAG_RE.sub('', text)
                    _open_at = text.rfind(_SESSION_GIST_OPEN)
                    if _open_at != -1 and _SESSION_GIST_CLOSE not in text[_open_at:]:
                        text = text[:_open_at]
                        _gist_open = True
                    elif not _gist_open:
                        # 태그가 청크 경계에서 쪼개진 경우 꼬리를 보류해 다음 청크와 합쳐 판정
                        _n = _gist_tag_partial_suffix_len(text)
                        if _n:
                            _gist_carry = text[-_n:]
                            text = text[:-_n]
                    text = text.strip()
                    # last_reasoning이 비어있으면 [알림:...] 내용으로만 채움.
                    # 우선순위: reasoning_message > [알림:...] 내용
                    # ※ assistant_message 본문 전체를 intent로 쓰지 않음:
                    #   본문은 응답 텍스트 전체라서 팝업에 표시하면 너무 길고 노이즈가 많음.
                    if not last_reasoning and _notify_texts:
                        last_reasoning = " | ".join(_notify_texts)
                        _save_reasoning(last_reasoning, self.conversation_id)
                    if not text:
                        continue
                    if _speak_only_final:
                        _pending_speech += text
                        continue
                    # 기본(집): 받은 대사를 그대로 흘려보낸다
                    _yielded += 1
                    yield text
                    continue
                elif token.message_type == "letta_error":
                    logger.error(f"[Letta] 에러 응답 수신: {getattr(token, 'error', token)}")
                    _stream_failed = True
                    continue
                elif token.message_type == "tool_call_message":
                    _tc = getattr(token, "tool_call", None)
                    tool_name = getattr(_tc, "name", None) or "unknown"
                    if _speak_only_final and tool_name != "unknown":
                        if tool_name in _MEMORY_WRITE_TOOLS:
                            if _pending_speech:
                                _deferred_final_speech = _pending_speech
                        else:
                            _deferred_final_speech = ""
                            if _pending_speech and _pending_speech not in _found_notifies:
                                yield {"type": "progress_update", "text": _pending_speech, "timestamp": _time.time()}
                                _found_notifies.add(_pending_speech)
                        _pending_speech = ""
                    elif not _speak_only_final:
                        # 기본(집): step 마다 알림 중복 방지 집합을 비운다
                        _found_notifies = set()
                    _had_tool_call = True
                    tool_id = getattr(_tc, "id", None) or f"tc_{int(_time.time()*1000)}"
                    # hook이 /tool-approval POST 전에 읽어갈 수 있도록 reasoning을 파일에 저장
                    _save_reasoning(last_reasoning, self.conversation_id)
                    # 세션 활동 상태를 툴 실행 중으로 업데이트 (프론트 스피너 텍스트 반영)
                    yield {"type": "agent_status", "status": "active", "detail": f"툴 실행 중 · {tool_name}", "timestamp": _time.time()}
                    yield {"type": "tool_call_status", "status": "running", "tool_name": tool_name, "tool_id": tool_id, "timestamp": _time.time(), "intent": last_reasoning}
                    # 다음 step을 위해 초기화
                    last_reasoning = ""
                    _assistant_buffer = ""
                    # 요약 태그 상태도 초기화 — 짝 안 맞는 태그가 다음 step 발화를 통째로
                    # 삼켜버리는 사고 방지(버퍼가 비워지므로 닫는 태그를 영원히 못 찾음)
                    _gist_open = False
                    _gist_carry = ""
                    _found_images = set()
                    _notify_file_offset = 0  # 다음 step의 알림 파일도 처음부터
                    continue
                elif token.message_type == "tool_return_message":
                    tool_name = getattr(token, "name", None) or "unknown"
                    tool_id = getattr(token, "tool_call_id", None) or getattr(token, "id", None) or f"tr_{int(_time.time()*1000)}"
                    status = getattr(token, "status", "success")
                    if status != "success":
                        _deferred_final_speech = ""
                    yield {"type": "tool_call_status", "status": status, "tool_name": tool_name, "tool_id": tool_id, "timestamp": _time.time()}
                    # 툴 반환 후 다시 응답 생성 단계로 전환
                    yield {"type": "agent_status", "status": "active", "detail": "생각 중", "timestamp": _time.time()}
                    continue
                elif token.message_type == "stop_reason":
                    _reason = getattr(token, "stop_reason", None)
                    _reason = getattr(_reason, "value", _reason)
                    if _reason not in (None, "end_turn"):
                        logger.warning(f"[Letta] 미완료 종료: {_reason}")
                        _stream_failed = True
                    continue
                elif token.message_type in ("usage_statistics", "ping"):
                    continue
                else:
                    logger.debug(f"[Letta] 알 수 없는 토큰 타입 무시: {token.message_type}")
                    continue

            if not self._abort_flag:
                if _speak_only_final:
                    if _stream_failed:
                        _yielded += 1
                        yield "작업이 완료되기 전에 중단됐습니다. 진행 내역을 확인해주세요."
                    elif _pending_speech or _deferred_final_speech:
                        _yielded += 1
                        yield _pending_speech or _deferred_final_speech
                    elif _had_tool_call:
                        _yielded += 1
                        yield "작업이 완료되기 전에 응답이 중단됐습니다. 진행 내역을 확인해주세요."
                elif _stream_failed and _yielded == 0:
                    # 기본(집)에서는 대사를 이미 다 흘려보냈으므로, 한 마디도 못 한
                    # 경우에만 중단 사실을 알린다. 정상 대사를 덮어쓰지 않는다.
                    _yielded += 1
                    yield "작업이 완료되기 전에 중단됐습니다. 진행 내역을 확인해주세요."

            # 스트림 종료 후 남은 ADB 스크린샷 마지막 flush
            while not _image_queue.empty():
                _pi = _image_queue.get_nowait()
                yield {"type": "image_display", "image_base64": _pi["image_base64"], "mime_type": "image/png", "label": _pi["label"]}

            # 미즈키가 남긴 세션 한 줄 요약 저장 (호버 툴팁에 쓰임)
            if _session_gist:
                try:
                    await self.update_session_gist(self.conversation_id, _session_gist)
                    if self._websocket_send_fn is not None:
                        await self._websocket_send_fn(json.dumps({
                            "type": "session-gist-updated",
                            "session_id": self.conversation_id,
                            "gist": _session_gist,
                        }))
                except Exception as _ge:
                    logger.debug(f"[session] 요약 저장 중 오류: {_ge}")
        finally:
            _polling_stop.set()
            _poll_task.cancel()
            _adb_poll_task.cancel()
            _stream_feed_task.cancel()
            try:
                await _poll_task
            except asyncio.CancelledError:
                pass
            try:
                await _adb_poll_task
            except asyncio.CancelledError:
                pass
            try:
                await _stream_feed_task
            except asyncio.CancelledError:
                pass
            # 세션 활동 idle 전송: generator 종료 경로에서도 확실히 스피너를 꺼야 하므로
            # 웹소켓 직접 전송을 시도한다. _websocket_send_fn이 없으면 조용히 스킵.
            if self._websocket_send_fn is not None:
                try:
                    await self._websocket_send_fn(json.dumps({
                        "type": "agent_status",
                        "status": "idle",
                        "timestamp": _time.time(),
                    }))
                except Exception as _idle_err:
                    logger.debug(f"[Letta] agent_status idle 전송 실패 (무시): {_idle_err}")

        if _yielded == 0:
            logger.warning("[Letta] 응답 토큰 없음 — LLM이 아무것도 반환하지 않았습니다. Letta 서버 로그를 확인하세요.")

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(f"[Clipboard content: {text_data.content}]")

        return "\n".join(message_parts)
