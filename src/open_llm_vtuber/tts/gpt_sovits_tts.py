####
# change from xTTS.py
####

import re
from typing import List, Optional

import requests
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """GPT-SoVITS API TTS 엔진.

    api_url(1순위)이 연결 불가일 때 fallback_api_urls를 순서대로 시도한다.
    - 연결 거부/끊김(ConnectionError)과 연결 타임아웃은 read 타임아웃을 기다리지 않고
      즉시 다음 후보로 넘어간다 (connect_timeout만 소비).
    - 폴백 상태는 보존하지 않는다. 매 요청마다 항상 1순위부터 다시 시도한다.
    """

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880/tts",
        text_lang: str = "zh",
        ref_audio_path: str = "",
        prompt_lang: str = "zh",
        prompt_text: str = "",
        text_split_method: str = "cut5",
        batch_size: str = "1",
        media_type: str = "wav",
        streaming_mode: str = "ture",
        fallback_api_urls: Optional[List[str]] = None,
        connect_timeout: float = 3.0,
        read_timeout: float = 600.0,
    ):
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.media_type = media_type
        self.streaming_mode = streaming_mode
        self.fallback_api_urls = [u for u in (fallback_api_urls or []) if u]
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        if self.fallback_api_urls:
            logger.info(
                f"[GSV-TTS] 폴백 주소 {len(self.fallback_api_urls)}개 등록: "
                f"{', '.join(self.fallback_api_urls)}"
            )

    def _candidate_urls(self) -> List[str]:
        """시도 순서대로 중복 제거한 후보 URL 목록."""
        urls: List[str] = []
        for url in [self.api_url, *self.fallback_api_urls]:
            if url and url not in urls:
                urls.append(url)
        return urls

    def generate_audio(self, text, file_name_no_ext=None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        cleaned_text = re.sub(r"\[.*?\]", "", text)
        # Prepare the data for the POST request
        data = {
            "text": cleaned_text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": self.prompt_lang,
            "prompt_text": self.prompt_text,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,
        }

        candidates = self._candidate_urls()
        last_error = "no candidate url"

        for idx, url in enumerate(candidates):
            is_fallback = idx > 0
            try:
                response = requests.get(
                    url,
                    params=data,
                    # (connect, read) — 연결 실패는 connect_timeout 안에 바로 드러난다
                    timeout=(self.connect_timeout, self.read_timeout),
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                # 연결 거부/끊김/타임아웃 → 서버가 죽은 것으로 보고 다음 후보로
                last_error = f"{type(e).__name__}: {e}"
                if idx + 1 < len(candidates):
                    logger.warning(
                        f"[GSV-TTS] {url} 연결 실패 ({type(e).__name__}) "
                        f"→ 폴백 {candidates[idx + 1]} 시도"
                    )
                else:
                    logger.error(f"[GSV-TTS] {url} 연결 실패 ({type(e).__name__}), 후보 없음")
                continue
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.error(f"[GSV-TTS] {url} 요청 오류: {last_error}")
                continue

            if response.status_code == 200:
                with open(file_name, "wb") as audio_file:
                    audio_file.write(response.content)
                if is_fallback:
                    logger.info(f"[GSV-TTS] 폴백 서버로 합성 성공: {url}")
                return file_name

            # 응답은 왔지만 실패 — 서버가 떠 있어도 합성이 안 되면 소용없으므로 다음 후보도 시도
            last_error = f"status {response.status_code}"
            body = ""
            try:
                body = response.text[:200]
            except Exception:
                pass
            if idx + 1 < len(candidates):
                logger.warning(
                    f"[GSV-TTS] {url} 합성 실패 (status {response.status_code}, {body!r}) "
                    f"→ 폴백 {candidates[idx + 1]} 시도"
                )
            else:
                logger.critical(
                    f"Error: Failed to generate audio. Status code: {response.status_code} "
                    f"({url}, {body!r})"
                )

        logger.critical(f"[GSV-TTS] 모든 후보 실패 ({len(candidates)}개). 마지막 오류: {last_error}")
        return None
