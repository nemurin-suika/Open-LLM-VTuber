import re
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        irodori_tts_path: str = "/Users/nemulin/develop/Irodori-TTS",
        checkpoint: Optional[str] = None,
        hf_checkpoint: str = "Aratako/Irodori-TTS-500M-v2",
        ref_wav: Optional[str] = None,
        no_ref: bool = False,
        model_device: str = "auto",
        model_precision: str = "fp32",
        codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
        seconds: float = 30.0,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_caption: float = 3.0,
        cfg_scale_speaker: float = 5.0,
        cfg_guidance_mode: str = "independent",
        seed: Optional[int] = None,
    ):
        irodori_path = str(Path(irodori_tts_path).expanduser())
        if irodori_path not in sys.path:
            sys.path.insert(0, irodori_path)

        from irodori_tts.inference_runtime import (
            InferenceRuntime,
            RuntimeKey,
            SamplingRequest,
            default_runtime_device,
            save_wav,
        )

        self._SamplingRequest = SamplingRequest
        self._save_wav = save_wav

        if model_device == "auto":
            model_device = default_runtime_device()

        if checkpoint is not None:
            checkpoint_path = str(Path(checkpoint).expanduser())
            if not Path(checkpoint_path).is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            from huggingface_hub import hf_hub_download

            logger.info(f"Downloading model from HuggingFace: {hf_checkpoint}")
            checkpoint_path = hf_hub_download(
                repo_id=hf_checkpoint,
                filename="model.safetensors",
            )

        self.ref_wav = ref_wav
        self.no_ref = no_ref
        self.seconds = seconds
        self.num_steps = num_steps
        self.cfg_scale_text = cfg_scale_text
        self.cfg_scale_caption = cfg_scale_caption
        self.cfg_scale_speaker = cfg_scale_speaker
        self.cfg_guidance_mode = cfg_guidance_mode
        self.seed = seed

        logger.info(f"Loading Irodori-TTS from {checkpoint_path} on {model_device}...")
        key = RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=model_device,
            codec_repo=codec_repo,
            model_precision=model_precision,
        )
        self.runtime = InferenceRuntime.from_key(key)
        logger.info("Irodori-TTS loaded.")

        Path("cache").mkdir(exist_ok=True)

    def generate_audio(self, text: str, file_name_no_ext=None) -> Optional[str]:
        cleaned_text = re.sub(r"\[.*?\]", "", text).strip()
        if not cleaned_text:
            logger.warning("Text is empty after cleaning, skipping synthesis.")
            return None

        file_name = self.generate_cache_file_name(file_name_no_ext, "wav")

        req = self._SamplingRequest(
            text=cleaned_text,
            ref_wav=self.ref_wav,
            no_ref=self.no_ref,
            seconds=self.seconds,
            num_steps=self.num_steps,
            cfg_scale_text=self.cfg_scale_text,
            cfg_scale_caption=self.cfg_scale_caption,
            cfg_scale_speaker=self.cfg_scale_speaker,
            cfg_guidance_mode=self.cfg_guidance_mode,
            seed=self.seed,
        )

        try:
            result = self.runtime.synthesize(req)
            self._save_wav(file_name, result.audio, result.sample_rate)
            return file_name
        except Exception as e:
            logger.critical(f"Irodori-TTS synthesis failed: {e}")
            return None
