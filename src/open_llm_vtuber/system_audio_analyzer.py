"""시스템 오디오 분석 모듈.

프론트엔드에서 캡처한 시스템 오디오(base64 WAV/WebM)를 받아
ai-watching-chzzk와 동일한 파이프라인으로 분석한다:
- analyze-audio 서버: Demucs 보컬 분리 + pyannote 화자 분리 + SER + BGM
- 로컬 faster-whisper: 전사

LLM에 전달할 텍스트 블록을 반환한다.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger


# conf.yaml의 system_config.analyze_audio_url 값을 set_analyze_audio_url()로 주입.
# 환경 변수 ANALYZE_AUDIO_URL이 설정되어 있으면 그게 최우선.
_ANALYZE_AUDIO_URL = os.environ.get(
    "ANALYZE_AUDIO_URL", "http://192.168.0.15:9881"
).rstrip("/")

# 1순위 서버가 연결 불가일 때 순서대로 시도할 예비 주소
# (conf.yaml의 system_config.analyze_audio_fallback_urls)
_ANALYZE_AUDIO_FALLBACK_URLS: list[str] = [
    u.strip().rstrip("/")
    for u in os.environ.get("ANALYZE_AUDIO_FALLBACK_URLS", "").split(",")
    if u.strip()
]

# 연결 실패(거부/끊김)는 이 시간 안에 드러난다. read 타임아웃까지 기다리지 않고 바로 폴백.
_CONNECT_TIMEOUT = 3.0

# 폴백 시 다음 후보로 넘어가는 연결 계층 예외
_CONNECT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
)


def set_analyze_audio_url(url: str, fallback_urls: Optional[list[str]] = None) -> None:
    """conf.yaml에서 읽은 analyze-audio URL을 모듈에 주입. 환경 변수가 있으면 무시."""
    global _ANALYZE_AUDIO_URL, _ANALYZE_AUDIO_FALLBACK_URLS
    if not os.environ.get("ANALYZE_AUDIO_URL"):
        if url:
            _ANALYZE_AUDIO_URL = url.rstrip("/")
    if not os.environ.get("ANALYZE_AUDIO_FALLBACK_URLS"):
        if fallback_urls is not None:
            _ANALYZE_AUDIO_FALLBACK_URLS = [
                u.rstrip("/") for u in fallback_urls if u
            ]


def _candidate_bases() -> list[str]:
    """시도 순서대로 중복 제거한 후보 베이스 URL 목록."""
    bases: list[str] = []
    for base in [_ANALYZE_AUDIO_URL, *_ANALYZE_AUDIO_FALLBACK_URLS]:
        if base and base not in bases:
            bases.append(base)
    return bases


async def _aa_post(
    client: httpx.AsyncClient,
    path: str,
    *,
    files: dict,
    data: Optional[dict] = None,
    timeout: float = 600.0,
) -> httpx.Response:
    """analyze-audio 서버에 POST. 1순위가 연결 불가면 예비 주소로 즉시 폴백.

    - 연결 거부/끊김/연결 타임아웃이면 다음 후보로 넘어간다.
    - 응답이 왔는데 상태 코드가 오류여도 다음 후보를 시도한다 (모델 로드 실패 등).
    - 폴백 상태는 보존하지 않는다. 매 요청마다 항상 1순위부터 다시 시도한다.
    - 모든 후보가 실패하면 마지막 예외를 그대로 올린다 (호출자가 로깅/무시 처리).
    """
    bases = _candidate_bases()
    last_exc: Optional[Exception] = None

    for idx, base in enumerate(bases):
        has_next = idx + 1 < len(bases)
        try:
            r = await client.post(
                f"{base}{path}",
                files=files,
                data=data,
                timeout=httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT),
            )
            r.raise_for_status()
            if idx > 0:
                logger.info(f"[SysAudio] 폴백 서버로 {path} 성공: {base}")
            return r
        except _CONNECT_ERRORS as e:
            last_exc = e
            if has_next:
                logger.warning(
                    f"[SysAudio] {base}{path} 연결 실패 ({type(e).__name__}) "
                    f"→ 폴백 {bases[idx + 1]} 시도"
                )
            continue
        except httpx.HTTPStatusError as e:
            last_exc = e
            if has_next:
                logger.warning(
                    f"[SysAudio] {base}{path} 응답 오류 (status "
                    f"{e.response.status_code}) → 폴백 {bases[idx + 1]} 시도"
                )
            continue

    raise last_exc if last_exc else RuntimeError("analyze-audio 후보 주소 없음")


@dataclass
class _Utterance:
    start: float
    end: float
    label: str  # 화자 라벨 (스트리머/화자2/미상)
    text: str
    emotion: Optional[str] = None
    intensity: Optional[float] = None


# ---------------------------------------------------------------------------
# 입력 → /tmp WAV 파일
# ---------------------------------------------------------------------------
def _decode_to_wav(b64_data: str, mime_type: str) -> Optional[Path]:
    """base64 오디오 → 16kHz mono WAV. ffmpeg으로 변환. 실패 시 None."""
    if b64_data.startswith("data:") and "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_data)
    except Exception as e:
        logger.warning(f"[SysAudio] base64 디코딩 실패: {e}")
        return None
    if len(raw) < 1000:
        logger.debug(f"[SysAudio] 오디오 데이터 너무 작음 ({len(raw)}B), 건너뜀")
        return None

    uid = uuid.uuid4().hex
    ext = "webm"
    if "wav" in mime_type:
        ext = "wav"
    elif "mp4" in mime_type or "m4a" in mime_type:
        ext = "m4a"
    elif "ogg" in mime_type:
        ext = "ogg"
    src_path = Path(f"/tmp/sysaudio_{uid}.{ext}")
    wav_path = Path(f"/tmp/sysaudio_{uid}.wav")
    try:
        src_path.write_bytes(raw)
        # MediaRecorder의 WebM 컨테이너는 듀레이션 헤더가 없어 일반 입력 검출이
        # 실패할 수 있으므로 입력 포맷을 mime에서 추론해 -f로 강제 지정.
        input_format = None
        if "webm" in mime_type:
            input_format = "webm"
        elif "ogg" in mime_type:
            input_format = "ogg"
        elif "mp4" in mime_type or "m4a" in mime_type:
            input_format = "mp4"
        elif "wav" in mime_type:
            input_format = "wav"

        cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if input_format:
            cmd += ["-f", input_format]
        cmd += [
            "-i", str(src_path),
            "-ac", "1", "-ar", "16000",
            "-vn", str(wav_path),
        ]
        # 오디오 길이가 최대 600초까지 늘어날 수 있어 변환 타임아웃도 함께 상향
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 1000:
            stderr_full = result.stderr.decode("utf-8", errors="replace").strip()
            logger.warning(
                f"[SysAudio] ffmpeg 변환 실패 (exit {result.returncode}) "
                f"src={src_path} ({len(raw)}B, mime={mime_type})\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stderr:\n{stderr_full}"
            )
            return None
        return wav_path
    except Exception as e:
        logger.warning(f"[SysAudio] WAV 변환 오류: {e}")
        return None
    finally:
        try:
            if src_path.exists():
                src_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# analyze-audio 서버 호출
# ---------------------------------------------------------------------------
async def _aa_separate(client: httpx.AsyncClient, wav_path: Path) -> Optional[Path]:
    """/separate → 보컬만 분리한 WAV 경로. 실패 시 None."""
    try:
        with open(wav_path, "rb") as f:
            r = await _aa_post(
                client,
                "/separate",
                files={"audio": (wav_path.name, f.read(), "audio/wav")},
                data={"stem": "vocals"},
                timeout=600.0,
            )
        out = wav_path.with_name(f"{wav_path.stem}_vocals.wav")
        out.write_bytes(r.content)
        return out
    except Exception as e:
        logger.warning(f"[SysAudio] /separate 실패 (무시): {e}")
        return None


async def _aa_diarize(client: httpx.AsyncClient, wav_path: Path) -> Optional[dict]:
    """/diarize → {segments, embeddings}. 실패 시 None."""
    try:
        with open(wav_path, "rb") as f:
            r = await _aa_post(
                client,
                "/diarize",
                files={"audio": (wav_path.name, f.read(), "audio/wav")},
                timeout=600.0,
            )
        return r.json()
    except Exception as e:
        logger.warning(f"[SysAudio] /diarize 실패 (무시): {e}")
        return None


async def _aa_ser(
    client: httpx.AsyncClient, wav_path: Path, segments: Optional[list[dict]] = None
) -> Optional[list]:
    """/ser → 세그먼트별 감정 리스트."""
    try:
        with open(wav_path, "rb") as f:
            files = {"audio": (wav_path.name, f.read(), "audio/wav")}
            data = None
            if segments:
                data = {
                    "segments": json.dumps(
                        [{"start": s["start"], "end": s["end"]} for s in segments]
                    )
                }
            r = await _aa_post(
                client, "/ser", files=files, data=data, timeout=600.0,
            )
        return r.json()
    except Exception as e:
        logger.warning(f"[SysAudio] /ser 실패 (무시): {e}")
        return None


async def _aa_bgm(client: httpx.AsyncClient, wav_path: Path) -> Optional[dict]:
    """/bgm_analyze → BGM 특징."""
    try:
        with open(wav_path, "rb") as f:
            r = await _aa_post(
                client,
                "/bgm_analyze",
                files={"audio": (wav_path.name, f.read(), "audio/wav")},
                timeout=300.0,
            )
        return r.json()
    except Exception as e:
        logger.warning(f"[SysAudio] /bgm_analyze 실패 (무시): {e}")
        return None


# ---------------------------------------------------------------------------
# 원격 Whisper 전사 (analyze-audio /transcribe)
# ---------------------------------------------------------------------------
async def _aa_transcribe(client: httpx.AsyncClient, wav_path: Path) -> list[dict]:
    """/transcribe → Whisper 세그먼트 리스트. 실패 시 빈 리스트.

    - language="" → Whisper 자동 감지 (게임/방송 영상은 영어/일본어가 흔함)
    - vad_filter=false → 짧은 클립(10s) + BGM 섞인 오디오에서 VAD가 발화 구간을 통째로 잘라내는 케이스 방지
    """
    try:
        with open(wav_path, "rb") as f:
            r = await _aa_post(
                client,
                "/transcribe",
                files={"audio": (wav_path.name, f.read(), "audio/wav")},
                data={"language": "", "vad_filter": "false"},
                timeout=600.0,
            )
        return r.json()
    except Exception as e:
        logger.warning(f"[SysAudio] /transcribe 실패 (무시): {e}")
        return []


# ---------------------------------------------------------------------------
# 발화 통합
# ---------------------------------------------------------------------------
def _build_utterances(
    diarize_segments: list[dict],
    whisper_segs: list[dict],
    ser_results: list,
) -> list[_Utterance]:
    """화자 segments + whisper text + SER 감정을 결합."""
    ser_by_start = {round(r["start"], 2): r for r in (ser_results or [])}
    result: list[_Utterance] = []

    if diarize_segments:
        # 화자별 라벨링: 발화 시간 최다 = 스트리머, 나머지는 화자2~
        durations: dict[str, float] = {}
        for seg in diarize_segments:
            durations[seg["speaker"]] = durations.get(seg["speaker"], 0.0) + (seg["end"] - seg["start"])
        sorted_speakers = sorted(durations.items(), key=lambda x: -x[1])
        labels: dict[str, str] = {}
        for i, (spk, _) in enumerate(sorted_speakers):
            labels[spk] = "스트리머" if i == 0 else f"화자{i + 1}"

        for spk in diarize_segments:
            texts = []
            for ws in whisper_segs:
                overlap = max(0.0, min(spk["end"], ws["end"]) - max(spk["start"], ws["start"]))
                ws_dur = max(0.01, ws["end"] - ws["start"])
                if overlap / ws_dur > 0.5:
                    texts.append(ws.get("text", "").strip())
            ser = ser_by_start.get(round(spk["start"], 2))
            result.append(_Utterance(
                start=spk["start"], end=spk["end"], label=labels.get(spk["speaker"], "미상"),
                text=" ".join(t for t in texts if t),
                emotion=ser["label"] if ser else None,
                intensity=ser["intensity"] if ser else None,
            ))

        # Fallback: diarize/whisper 시간이 안 맞아서 매칭된 텍스트가 0개라면
        # whisper segments를 단일 화자로 직접 사용 (대사는 살아남게)
        if whisper_segs and not any(u.text for u in result):
            logger.info(
                f"[SysAudio] diarize-whisper 매칭 실패 ({len(diarize_segments)}/{len(whisper_segs)}) "
                "→ whisper segments fallback"
            )
            result = [
                _Utterance(
                    start=ws["start"], end=ws["end"], label="음성",
                    text=ws.get("text", "").strip(),
                    emotion=None, intensity=None,
                )
                for ws in whisper_segs
            ]
    else:
        # 화자 분리 없음 → 모든 whisper 발화를 단일 화자로
        for ws in whisper_segs:
            ser = ser_by_start.get(round(ws["start"], 2))
            result.append(_Utterance(
                start=ws["start"], end=ws["end"], label="음성",
                text=ws.get("text", "").strip(),
                emotion=ser["label"] if ser else None,
                intensity=ser["intensity"] if ser else None,
            ))
    return result


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
def _format_for_llm(utterances: list[_Utterance], bgm: Optional[dict]) -> str:
    """분석 결과를 [시스템 오디오] 태그 안에 들어갈 텍스트로 포맷.

    프롬프트의 [음성 ...] / [BGM ...] 라인 형식과 정확히 맞춤.
    """
    lines: list[str] = []
    for u in utterances:
        if not u.text:
            continue
        emo = f" {u.emotion}/{u.intensity:.2f}" if u.emotion else ""
        lines.append(f"[음성 {u.label} {u.start:.1f}~{u.end:.1f}s{emo}]: {u.text}")
    if bgm:
        tempo = bgm.get("tempo", 0)
        key = bgm.get("key", "?")
        mood = bgm.get("mood", "")
        duration = bgm.get("duration", 0)
        lines.append(f"[BGM 0.0~{duration:.1f}s {tempo:.0f}BPM, {key}, {mood}]")
    if not lines:
        return ""
    return "\n".join(lines)


async def analyze_system_audio(b64_data: str, mime_type: str = "audio/webm") -> str:
    """프론트엔드에서 받은 base64 오디오를 분석해서 [시스템 오디오] 블록 텍스트를 반환.

    빈 문자열을 반환하면 분석할 게 없거나 실패한 것이므로 호출자는 태그를 추가하지 말 것.
    """
    t_start = time.monotonic()
    wav_path = await asyncio.to_thread(_decode_to_wav, b64_data, mime_type)
    if not wav_path:
        return ""

    # 16kHz mono 16bit 기준 32000 B/s → WAV 크기로 대략적인 길이를 역산
    try:
        audio_seconds = max(0.0, (wav_path.stat().st_size - 44) / 32000.0)
    except Exception:
        audio_seconds = 0.0
    logger.info(
        f"[SysAudio] 분석 시작: 오디오 약 {audio_seconds:.0f}초, "
        f"디코딩 {time.monotonic() - t_start:.1f}s"
    )

    vocals_path: Optional[Path] = None
    try:
        async with httpx.AsyncClient() as client:
            # 1. 보컬 분리 (실패해도 원본으로 계속)
            t_step = time.monotonic()
            vocals_path = await _aa_separate(client, wav_path)
            target = vocals_path or wav_path
            logger.info(
                f"[SysAudio] 분리 결과: vocals={'있음' if vocals_path else '없음'}, "
                f"target={target.name}, {time.monotonic() - t_step:.1f}s"
            )

            # 2. Whisper 전사: 원본으로 먼저 시도. 비어있고 vocals가 있으면 vocals로 재시도.
            t_step = time.monotonic()
            whisper_segs = await _aa_transcribe(client, wav_path)
            logger.info(
                f"[SysAudio] Whisper (원본) segments={len(whisper_segs)}, "
                f"first_text={(whisper_segs[0].get('text', '')[:80] if whisper_segs else '')!r}, "
                f"{time.monotonic() - t_step:.1f}s"
            )
            if not whisper_segs and vocals_path:
                whisper_segs = await _aa_transcribe(client, vocals_path)
                logger.info(
                    f"[SysAudio] Whisper (vocals fallback) segments={len(whisper_segs)}, "
                    f"first_text={(whisper_segs[0].get('text', '')[:80] if whisper_segs else '')!r}"
                )

            # 3. 화자 분리 (vocals 우선 — 발화 구간 정렬에 더 정확)
            t_step = time.monotonic()
            diarize_data = await _aa_diarize(client, target)
            diarize_segments = (diarize_data or {}).get("segments", []) if diarize_data else []
            logger.info(
                f"[SysAudio] Diarize segments={len(diarize_segments)}, "
                f"{time.monotonic() - t_step:.1f}s"
            )

            # 4. SER
            t_step = time.monotonic()
            ser_segs = diarize_segments or whisper_segs
            ser_results = await _aa_ser(client, target, ser_segs) if ser_segs else None
            logger.info(
                f"[SysAudio] SER results={len(ser_results) if ser_results else 0}, "
                f"{time.monotonic() - t_step:.1f}s"
            )

            # 5. BGM (원본 오디오)
            t_step = time.monotonic()
            bgm = await _aa_bgm(client, wav_path)
            logger.info(f"[SysAudio] BGM 분석 {time.monotonic() - t_step:.1f}s")

        utterances = _build_utterances(diarize_segments, whisper_segs, ser_results or [])
        non_empty = sum(1 for u in utterances if u.text)
        logger.info(
            f"[SysAudio] _build_utterances: 총 {len(utterances)}개 (텍스트 있는 것 {non_empty}개)"
        )
        text_block = _format_for_llm(utterances, bgm)
        if text_block:
            logger.info(
                f"[SysAudio] 분석 완료: 발화 {non_empty}개, "
                f"BGM={'있음' if bgm else '없음'}, 텍스트={len(text_block)}자, "
                f"오디오 {audio_seconds:.0f}초 / 총 소요 {time.monotonic() - t_start:.1f}s"
            )
        else:
            logger.warning(
                f"[SysAudio] _format_for_llm 결과 비어있음 (utterances={len(utterances)}, BGM={'있음' if bgm else '없음'})"
            )
        return text_block
    except Exception as e:
        logger.warning(f"[SysAudio] 분석 오류: {e}")
        return ""
    finally:
        for p in (wav_path, vocals_path):
            if p and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
