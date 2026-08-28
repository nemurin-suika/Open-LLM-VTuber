import re
from typing import AsyncIterator, Tuple, Callable, List, Union, Dict, Any
from functools import wraps
from .output_types import Actions, SentenceOutput, DisplayText
from ..utils.tts_preprocessor import tts_filter as filter_text
from ..live2d_model import Live2dModel
from ..config_manager import TTSPreprocessorConfig
from ..utils.sentence_divider import SentenceDivider
from ..utils.sentence_divider import SentenceWithTags, TagState
from loguru import logger


# TTS 억제 구간 태그. [무음시작] ... [무음끝] 사이는 자막·Live2D에는 그대로 남기고
# 음성만 읽지 않는다. 링크나 경로처럼 소리내면 곤란한 내용, 방해 금지 상황에 사용.
# 구간은 문장 경계를 넘어 이어질 수 있으므로 스트림 단위로 상태를 유지한다.
_SILENT_TAG_RE = re.compile(r"\[\s*무음\s*(시작|끝)\s*\]")


def _strip_silent_segments(text: str, silent: bool) -> Tuple[str, bool]:
    """무음 구간을 제거한 발화용 텍스트와, 다음 문장으로 이어질 무음 상태를 반환."""
    parts: List[str] = []
    pos = 0
    for m in _SILENT_TAG_RE.finditer(text):
        if not silent:
            parts.append(text[pos : m.start()])
        silent = m.group(1) == "시작"
        pos = m.end()
    if not silent:
        parts.append(text[pos:])
    return "".join(parts), silent


def sentence_divider(
    faster_first_response: bool = True,
    segment_method: str = "pysbd",
    valid_tags: List[str] = None,
):
    """
    Decorator that transforms token stream into sentences with tags

    Args:
        faster_first_response: bool - Whether to enable faster first response
        segment_method: str - Method for sentence segmentation
        valid_tags: List[str] - List of valid tags to process
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Union[str, Dict[str, Any]]]
        ],  # Expects str or dict
    ) -> Callable[
        ..., AsyncIterator[Union[SentenceWithTags, Dict[str, Any]]]
    ]:  # Yields SentenceWithTags or dict
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Union[SentenceWithTags, Dict[str, Any]]]:
            divider = SentenceDivider(
                faster_first_response=faster_first_response,
                segment_method=segment_method,
                valid_tags=valid_tags or [],
            )
            stream_from_func = func(*args, **kwargs)

            # Process the mixed stream using the updated SentenceDivider
            async for item in divider.process_stream(stream_from_func):
                if isinstance(item, SentenceWithTags):
                    logger.debug(f"sentence_divider yielding sentence: {item}")
                elif isinstance(item, dict):
                    logger.debug(f"sentence_divider yielding dict: {item}")
                yield item  # Yield either SentenceWithTags or dict
            # Flushing is handled within divider.process_stream

        return wrapper

    return decorator


def actions_extractor(live2d_model: Live2dModel):
    """
    Decorator that extracts actions from sentences, passing through dicts.
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Union[SentenceWithTags, Dict[str, Any]]]
        ],  # Input type hint
    ) -> Callable[
        ..., AsyncIterator[Union[Tuple[SentenceWithTags, Actions], Dict[str, Any]]]
    ]:  # Output type hint
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[
            Union[Tuple[SentenceWithTags, Actions], Dict[str, Any]]
        ]:  # Yield type hint
            stream = func(*args, **kwargs)
            async for item in stream:
                if isinstance(item, SentenceWithTags):
                    sentence = item
                    actions = Actions()
                    # Only extract emotions/gaze for non-tag text
                    if not any(
                        tag.state in [TagState.START, TagState.END]
                        for tag in sentence.tags
                    ):
                        expressions = live2d_model.extract_emotion(sentence.text)
                        if expressions:
                            actions.expressions = expressions
                        gaze = live2d_model.extract_gaze(sentence.text)
                        if gaze:
                            actions.gaze = gaze
                        movement = live2d_model.extract_movement(sentence.text)
                        if movement:
                            actions.movement = movement
                        model_scale = live2d_model.extract_model_scale(sentence.text)
                        if model_scale is not None:
                            actions.model_scale = model_scale
                        volume_adjustment = live2d_model.extract_volume_adjustment(sentence.text)
                        if volume_adjustment is not None:
                            actions.volume_adjustment = volume_adjustment
                            logger.debug(f"[volume] extracted {volume_adjustment:+d} from: {sentence.text!r}")
                    yield sentence, actions  # Yield the tuple
                elif isinstance(item, dict):
                    # Pass through dictionaries
                    yield item
                else:
                    logger.warning(
                        f"actions_extractor received unexpected type: {type(item)}"
                    )

        return wrapper

    return decorator


def display_processor():
    """
    Decorator that processes text for display, passing through dicts.
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Union[Tuple[SentenceWithTags, Actions], Dict[str, Any]]]
        ],  # Input type hint
    ) -> Callable[
        ...,
        AsyncIterator[
            Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
        ],
    ]:  # Output type hint
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[
            Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
        ]:  # Yield type hint
            stream = func(*args, **kwargs)

            async for item in stream:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], SentenceWithTags)
                ):
                    sentence, actions = item
                    text = sentence.text
                    # Handle think tag states
                    for tag in sentence.tags:
                        if tag.name == "think":
                            if tag.state == TagState.START:
                                text = "("
                            elif tag.state == TagState.END:
                                text = ")"

                    display = DisplayText(text=text)  # Simplified DisplayText creation
                    yield sentence, display, actions  # Yield the tuple
                elif isinstance(item, dict):
                    # Pass through dictionaries
                    yield item
                else:
                    logger.warning(
                        f"display_processor received unexpected type: {type(item)}"
                    )

        return wrapper

    return decorator


def tts_filter(
    tts_preprocessor_config: TTSPreprocessorConfig = None,
):
    """
    Decorator that filters text for TTS, passing through dicts.
    Skips TTS for think tag content.
    """

    def decorator(
        func: Callable[
            ...,
            AsyncIterator[
                Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
            ],
        ],  # Input type hint
    ) -> Callable[
        ..., AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]
    ]:  # Output type hint
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:  # Yield type hint
            stream = func(*args, **kwargs)
            config = tts_preprocessor_config or TTSPreprocessorConfig()
            # [무음시작]/[무음끝] 구간 상태 (문장을 넘어 이어짐)
            silent = False

            async for item in stream:
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and isinstance(item[1], DisplayText)
                ):
                    sentence, display, actions = item
                    # 무음 구간을 걷어낸 발화용 텍스트를 먼저 만든다
                    speakable, silent = _strip_silent_segments(display.text, silent)
                    # 자막에서는 태그 문자열만 지우고 내용은 그대로 보여준다
                    display.text = _SILENT_TAG_RE.sub("", display.text)
                    if any(tag.name == "think" for tag in sentence.tags):
                        tts = ""
                    elif not speakable.strip():
                        tts = ""
                    else:
                        tts = filter_text(
                            text=speakable,
                            remove_special_char=config.remove_special_char,
                            ignore_brackets=config.ignore_brackets,
                            ignore_parentheses=config.ignore_parentheses,
                            ignore_asterisks=config.ignore_asterisks,
                            ignore_angle_brackets=config.ignore_angle_brackets,
                        )

                    logger.debug(f"[{display.name}] display: {display.text}")
                    logger.debug(f"[{display.name}] tts: {tts}")

                    yield SentenceOutput(
                        display_text=display,
                        tts_text=tts,
                        actions=actions,
                    )
                elif isinstance(item, dict):
                    # Pass through dictionaries
                    yield item
                else:
                    logger.warning(f"tts_filter received unexpected type: {type(item)}")

        return wrapper

    return decorator
