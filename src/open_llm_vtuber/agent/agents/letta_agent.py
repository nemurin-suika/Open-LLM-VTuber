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
# ADB 스크린샷 감시: 미즈키가 ADB로 찍은 스크린샷을 프론트에 자동 전송
_ADB_SCREENSHOT_FILE = "/tmp/ihwan_screen.png"
# 팝업 표시 전 제거할 Live2D 태그 패턴 (표정/시선/볼륨)
_LIVE2D_TAG_RE = re.compile(
    r'\[(look_\w+|화남|역겨움|즐거움|슬픔|부끄러움|혼란함|volume_up|volume_down)\]'
)


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


def _save_reasoning(text: str) -> None:
    """last_reasoning에서 툴 승인 팝업용 intent를 추출해 파일 저장.

    추출 우선순위:
    1. {의도} 패턴 — 마지막 중괄호 안 텍스트 (툴 승인 팝업 전용 포맷)
    2. [알림: ...] 패턴 — 마지막 알림 (fallback, {} 없을 때만)
    3. 둘 다 없으면 저장 안 함
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
    try:
        with open(_REASONING_FILE, "w", encoding="utf-8") as _rf:
            json.dump({"text": intent, "ts": _time.time()}, _rf, ensure_ascii=False)
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
        else:
            # conf.yaml에 id가 박혀있어 _select_or_create_agent를 거치지 않는 경우에도,
            # 서버 시작 시 한 번 system_prompt diff를 체크해서 변경되었으면 자동 동기화.
            _sync_system_prompt_if_changed(self.url, id, system_prompt)

        self.client = AsyncLetta(base_url=self.url, timeout=600)
        self.id = id
        # 인터럽트 플래그: True이면 _chat_raw 스트림 루프를 조기 종료
        self._abort_flag: bool = False
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
        """인터럽트 신호 수신 시 _chat_raw 스트림 루프와 letta 서버의 claude -p subprocess를 종료.

        1. _abort_flag = True → 스트리밍 루프 조기 break
        2. abort 파일 생성 → letta 서버의 claude_cli_client.py watcher가 감지해서 proc.kill()
        """
        self._abort_flag = True
        try:
            with open(_ABORT_FLAG_FILE, "w", encoding="utf-8") as _af:
                _af.write(str(_time.time()))
            logger.info(f"[LettaAgent] abort 파일 생성: {_ABORT_FLAG_FILE}")
        except Exception as _e:
            logger.warning(f"[LettaAgent] abort 파일 생성 실패 (무시): {_e}")
        logger.info("[LettaAgent] 인터럽트 신호 수신 → _abort_flag + abort 파일 설정")

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
            if os.path.exists(_ABORT_FLAG_FILE):
                os.unlink(_ABORT_FLAG_FILE)
        except Exception:
            pass
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
                temp_path = f"/tmp/claude_screenshot_{_uuid.uuid4().hex}.jpg"
                await asyncio.to_thread(
                    lambda p=temp_path, d=img_data: open(p, "wb").write(_base64.b64decode(d))
                )
                # ago_seconds가 있으면 태그 속성으로 추가. 0이면 "현재", 양수면 "N초 전".
                ago = getattr(img, "ago_seconds", None)
                if ago is not None:
                    ago_attr = f" ago={ago:g}s"
                else:
                    ago_attr = ""
                screenshot_tags.append(f"[스크린샷 source={source}{ago_attr}]{temp_path}[/스크린샷]")
                logger.info(f"[Screenshot] temp 저장: {temp_path} (source={source}{ago_attr})")
                # ── 스크린샷도 프론트에 전송 ──
                mime = getattr(img, "media_type", None) or getattr(img, "mime_type", None) or "image/jpeg"
                yield {"type": "image_display", "image_base64": img_data, "mime_type": mime, "label": source}
            text = "\n".join(screenshot_tags) + "\n" + text
            input_content = text
        else:
            input_content = text

        stream = await self.client.agents.messages.create(
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
        _found_notifies: set = set()  # 이미 yield한 [알림:] 내용 — 중복 방지

        # -----------------------------------------------------------------------
        # 중간 진행 알림 실시간 표시:
        # claude_cli_client.py가 --output-format stream-json으로 실행하면서
        # [알림: ...] 감지 즉시 _PROGRESS_NOTIFY_FILE에 JSONL로 기록한다.
        # 이 polling task가 0.3초마다 파일을 읽어 _notify_queue에 넣고,
        # Letta 스트림 루프에서 pending 알림을 먼저 yield한다.
        # -----------------------------------------------------------------------
        _notify_queue: asyncio.Queue = asyncio.Queue()
        _image_queue: asyncio.Queue = asyncio.Queue()  # ADB 스크린샷 전송용
        _polling_stop = asyncio.Event()
        _notify_file_offset: int = 0
        # 스트림 시작 시점의 mtime으로 초기화 → 이전에 찍어둔 스크린샷 오발송 방지
        try:
            _adb_screenshot_mtime: float = os.path.getmtime(_ADB_SCREENSHOT_FILE)
        except FileNotFoundError:
            _adb_screenshot_mtime = 0.0

        async def _poll_progress_file() -> None:
            """백그라운드: 알림 파일을 0.3초마다 polling."""
            nonlocal _notify_file_offset
            while not _polling_stop.is_set():
                try:
                    with open(_PROGRESS_NOTIFY_FILE, "r", encoding="utf-8") as _pf:
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
            """백그라운드: ADB 스크린샷 파일 mtime 감시 → 변경 시 image_queue에 추가."""
            nonlocal _adb_screenshot_mtime
            while not _polling_stop.is_set():
                try:
                    mtime = os.path.getmtime(_ADB_SCREENSHOT_FILE)
                    if mtime > _adb_screenshot_mtime:
                        _adb_screenshot_mtime = mtime
                        img_bytes = await asyncio.to_thread(
                            lambda: open(_ADB_SCREENSHOT_FILE, "rb").read()
                        )
                        img_b64 = _base64.b64encode(img_bytes).decode()
                        await _image_queue.put({"image_base64": img_b64, "label": "adb"})
                        logger.debug(f"[Letta] ADB 스크린샷 변경 감지 → image_queue 추가")
                except FileNotFoundError:
                    pass
                except Exception as _ae:
                    logger.debug(f"[Letta] ADB 스크린샷 polling 오류 (무시): {_ae}")
                await asyncio.sleep(0.5)

        _poll_task = asyncio.create_task(_poll_progress_file())
        _adb_poll_task = asyncio.create_task(_poll_adb_screenshot())

        async def _drain_notify_queue() -> None:
            """pending 중간 알림을 모두 yield (호출자 generator에서 inline으로 처리)."""
            # 참고: generator 내부에서 다른 generator를 yield from 할 수 없으므로
            # 이 함수는 직접 호출하지 않고, 루프 안에서 while+get_nowait로 처리한다.
            pass

        try:
            async for token in stream:
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
                    _save_reasoning(last_reasoning)
                    continue
                elif token.message_type == "assistant_message":
                    content = token.content
                    if isinstance(content, list):
                        text = "".join(part.text for part in content)
                    else:
                        text = content
                    # 누적 버퍼에 추가 — 토큰 경계를 넘어 [알림: ...] 감지
                    _assistant_buffer += text
                    _notify_texts: list[str] = []
                    for _m in re.finditer(r'\[알림:\s*(.*?)\]', _assistant_buffer, re.DOTALL):
                        _notify = _m.group(1).strip()
                        if _notify and re.sub(r'[.\s…]+', '', _notify) and _notify not in _found_notifies:
                            yield {"type": "progress_update", "text": _notify, "timestamp": _time.time()}
                            _notify_texts.append(_notify)
                            _found_notifies.add(_notify)
                    # [알림: ...] 태그를 현재 토큰의 TTS/표시 텍스트에서 제거
                    text = re.sub(r'\[알림:\s*.*?\]', '', text, flags=re.DOTALL).strip()
                    # last_reasoning이 비어있으면 [알림:...] 내용으로만 채움.
                    # 우선순위: reasoning_message > [알림:...] 내용
                    # ※ assistant_message 본문 전체를 intent로 쓰지 않음:
                    #   본문은 응답 텍스트 전체라서 팝업에 표시하면 너무 길고 노이즈가 많음.
                    if not last_reasoning and _notify_texts:
                        last_reasoning = " | ".join(_notify_texts)
                        _save_reasoning(last_reasoning)
                    if not text:
                        continue
                elif token.message_type == "letta_error":
                    logger.error(f"[Letta] 에러 응답 수신: {getattr(token, 'error', token)}")
                    continue
                elif token.message_type == "tool_call_message":
                    _tc = getattr(token, "tool_call", None)
                    tool_name = getattr(_tc, "name", None) or "unknown"
                    tool_id = getattr(_tc, "id", None) or f"tc_{int(_time.time()*1000)}"
                    # hook이 /tool-approval POST 전에 읽어갈 수 있도록 reasoning을 파일에 저장
                    _save_reasoning(last_reasoning)
                    yield {"type": "tool_call_status", "status": "running", "tool_name": tool_name, "tool_id": tool_id, "timestamp": _time.time(), "intent": last_reasoning}
                    # 다음 step을 위해 초기화
                    last_reasoning = ""
                    _assistant_buffer = ""
                    _found_notifies = set()
                    _notify_file_offset = 0  # 다음 step의 알림 파일도 처음부터
                    continue
                elif token.message_type == "tool_return_message":
                    tool_name = getattr(token, "name", None) or "unknown"
                    tool_id = getattr(token, "tool_call_id", None) or getattr(token, "id", None) or f"tr_{int(_time.time()*1000)}"
                    status = getattr(token, "status", "success")
                    yield {"type": "tool_call_status", "status": status, "tool_name": tool_name, "tool_id": tool_id, "timestamp": _time.time()}
                    continue
                elif token.message_type in ("stop_reason", "usage_statistics", "ping"):
                    continue
                else:
                    logger.debug(f"[Letta] 알 수 없는 토큰 타입 무시: {token.message_type}")
                    continue

                _yielded += 1
                yield text

            # 스트림 종료 후 남은 ADB 스크린샷 마지막 flush
            while not _image_queue.empty():
                _pi = _image_queue.get_nowait()
                yield {"type": "image_display", "image_base64": _pi["image_base64"], "mime_type": "image/png", "label": _pi["label"]}
        finally:
            _polling_stop.set()
            _poll_task.cancel()
            _adb_poll_task.cancel()
            try:
                await _poll_task
            except asyncio.CancelledError:
                pass
            try:
                await _adb_poll_task
            except asyncio.CancelledError:
                pass

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
