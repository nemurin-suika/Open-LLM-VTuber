import json
import os
import re
import struct
from typing import Dict, Optional, Tuple

import chardet
import numpy as np
from loguru import logger

# Gaze direction presets: x = left(-1) to right(+1), y = down(-1) to up(+1)
GAZE_KEYWORDS: Dict[str, Dict[str, float]] = {
    "look_left": {"x": -0.7, "y": 0.0},
    "look_right": {"x": 0.7, "y": 0.0},
    "look_up": {"x": 0.0, "y": 0.7},
    "look_down": {"x": 0.0, "y": -0.7},
    "look_center": {"x": 0.0, "y": 0.0},
    "look_upleft": {"x": -0.7, "y": 0.7},
    "look_upright": {"x": 0.7, "y": 0.7},
    "look_downleft": {"x": -0.7, "y": -0.7},
    "look_downright": {"x": 0.7, "y": -0.7},
}

# Model body-movement presets — relative delta per tag (NDC units).
# Accumulates across multiple tags in the same sentence.
# "move_center" is special: sets absolute target (0, 0).
MOVEMENT_KEYWORDS: Dict[str, Dict[str, float]] = {
    "move_left":      {"dx": -0.45, "dy":  0.0},
    "move_right":     {"dx":  0.45, "dy":  0.0},
    "move_up":        {"dx":  0.0,  "dy":  0.3},
    "move_down":      {"dx":  0.0,  "dy": -0.3},
    "move_center":    {"abs": 1.0,  "dx":  0.0, "dy": 0.0},  # absolute reset
    "move_upleft":    {"dx": -0.45, "dy":  0.3},
    "move_upright":   {"dx":  0.45, "dy":  0.3},
    "move_downleft":  {"dx": -0.45, "dy": -0.3},
    "move_downright": {"dx":  0.45, "dy": -0.3},
}

# Scale multipliers relative to the model's reference (kScale) size
SCALE_KEYWORDS: Dict[str, float] = {
    "scale_up":     1.3,
    "scale_down":   0.75,
    "scale_normal": 1.0,
}

# Volume adjustment per keyword (percentage points, 0–100 scale)
VOLUME_KEYWORDS: Dict[str, int] = {
    "volume_up":   5,
    "volume_down": -5,
}

# This class will only prepare the payload for the live2d model
# the process of sending the payload should be done by the caller
# This class is **Not responsible** for sending the payload to the server


class Live2dModel:
    """
    A class to represent a Live2D model. This class only prepares and stores the information of the Live2D model. It does not send anything to the frontend or server or anything.

    Attributes:
        model_dict_path (str): The path to the model dictionary file.
        live2d_model_name (str): The name of the Live2D model.
        model_info (dict): The information of the Live2D model.
        emo_map (dict): The emotion map of the Live2D model.
        emo_str (str): The string representation of the emotion map of the Live2D model.
    """

    model_dict_path: str
    live2d_model_name: str
    model_info: dict
    emo_map: dict
    emo_str: str

    def __init__(
        self, live2d_model_name: str, model_dict_path: str = "model_dict.json"
    ):
        self.model_dict_path: str = model_dict_path
        self.live2d_model_name: str = live2d_model_name
        self.set_model(live2d_model_name)

    # Dynamic screen position updated by frontend in real-time.
    # Normalized coords (0-1): 0,0=top-left, 1,1=bottom-right.
    # Falls back to model_dict.json "screenPosition" or (0.85, 0.85) if not set.
    _dynamic_screen_position: Optional[Dict[str, float]] = None
    _model_screen_height: float = 0.5  # normalized model height on screen

    # Default eye position ratios used when model_dict.json has no matching field.
    _DEFAULT_EYE_TOP_RATIO: float = 0.2   # ratio from top   (0=top,  1=bottom)
    _DEFAULT_EYE_LEFT_RATIO: float = 0.5  # ratio from left  (0=left, 1=right)

    def update_screen_position(self, x: float, y: float, height: float = 0.5) -> None:
        """Update VTuber's current screen position and model height reported by the frontend."""
        self._dynamic_screen_position = {"x": float(x), "y": float(y)}
        self._model_screen_height = float(height)

    def _get_eye_screen_position(self) -> Dict[str, float]:
        """Return the screen position of the model's eyes (not its NDC center).

        The frontend reports the model's NDC origin (center of the matrix).
        Eyes may be offset from the origin both vertically and horizontally.

        eyeTopRatio  (0=top,  1=bottom, default 0.2): shifts y upward.
        eyeLeftRatio (0=left, 1=right,  default 0.5): shifts x left/right.
        """
        if self._dynamic_screen_position is not None:
            center_x = self._dynamic_screen_position["x"]
            center_y = self._dynamic_screen_position["y"]
        else:
            sp = self.model_info.get("screenPosition", {"x": 0.85, "y": 0.85})
            center_x = float(sp.get("x", 0.85))
            center_y = float(sp.get("y", 0.85))

        # eyeTopRatio: where the eyes are vertically (0=top, 1=bottom of model).
        # 0.5 = model center; smaller values shift eye reference upward on screen.
        eye_top_ratio = float(
            self.model_info.get("eyeTopRatio", self._DEFAULT_EYE_TOP_RATIO)
        )
        eye_y = center_y - (0.5 - eye_top_ratio) * self._model_screen_height

        # eyeLeftRatio: where the eyes are horizontally (0=left, 1=right of model).
        # 0.5 = model center; smaller values shift eye reference leftward on screen.
        eye_left_ratio = float(
            self.model_info.get("eyeLeftRatio", self._DEFAULT_EYE_LEFT_RATIO)
        )
        eye_x = center_x + (eye_left_ratio - 0.5) * self._model_screen_height

        return {"x": eye_x, "y": eye_y}

    def set_model(self, model_name: str) -> None:
        """
        Set the model with its name and load the model information. This method will initialize the `self.model_info`, `self.emo_map`, and `self.emo_str` attributes.
        This method is called in the constructor.

        Parameters:
            model_name (str): The name of the live2d model.

            Returns:
            None
        """

        self.model_info: dict = self._lookup_model_info(model_name)
        self.emo_map: dict = {
            k.lower(): v for k, v in self.model_info["emotionMap"].items()
        }
        self.emo_str: str = " ".join([f"[{key}]," for key in self.emo_map.keys()])
        # emo_str is a string of the keys in the emoMap dictionary. The keys are enclosed in square brackets.
        # example: `"[fear], [anger], [disgust], [sadness], [joy], [neutral], [surprise]"`

        # Auto-detect eye ratios from the moc3 if not set in model_dict.json
        self._auto_detect_eye_ratios_if_missing()

    # ------------------------------------------------------------------
    # Eye-ratio auto-detection from moc3
    # ------------------------------------------------------------------

    def _auto_detect_eye_ratios_if_missing(self) -> None:
        """Detect eyeTopRatio / eyeLeftRatio from the moc3 if absent in model_info."""
        needs_top = "eyeTopRatio" not in self.model_info
        needs_left = "eyeLeftRatio" not in self.model_info
        if not (needs_top or needs_left):
            return

        try:
            top, left = self._detect_eye_ratios_from_moc3()
            if needs_top:
                self.model_info["eyeTopRatio"] = top
                logger.info(f"Auto-detected eyeTopRatio={top:.3f} from moc3")
            if needs_left:
                self.model_info["eyeLeftRatio"] = left
                logger.info(f"Auto-detected eyeLeftRatio={left:.3f} from moc3")
        except Exception as e:
            logger.warning(f"Eye ratio auto-detection failed: {e}. Using defaults.")

    def _resolve_moc3_path(self) -> str:
        """Return the absolute path to the model's .moc3 file."""
        url: str = self.model_info.get("url", "")
        # URL may start with "/" (web-relative); strip it for local FS access
        model3_path = url.lstrip("/")
        if not os.path.isfile(model3_path):
            raise FileNotFoundError(f"model3.json not found at {model3_path!r}")

        raw = open(model3_path, "rb").read()
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        model3 = json.loads(raw.decode(enc, errors="replace"))

        moc_rel = model3.get("FileReferences", {}).get("Moc", "")
        if not moc_rel:
            raise ValueError("No Moc reference in model3.json")

        moc3_path = os.path.join(os.path.dirname(model3_path), moc_rel)
        if not os.path.isfile(moc3_path):
            raise FileNotFoundError(f"moc3 file not found at {moc3_path!r}")
        return moc3_path

    def _detect_eye_ratios_from_moc3(self) -> Tuple[float, float]:
        """
        Parse the moc3 binary to estimate where the character's eyes are located,
        then return (eyeTopRatio, eyeLeftRatio) as canvas-fraction values.

        The moc3 stores all ArtMesh vertex positions in pixel space.  We read
        the default-pose positions, convert them to canvas fractions (0=top-left,
        1=bottom-right), and find the y-band whose horizontal variance is highest
        (two eye clusters create the widest spread in that axis).

        Returns:
            (eyeTopRatio, eyeLeftRatio): both in [0, 1]
        """
        moc3_path = self._resolve_moc3_path()
        with open(moc3_path, "rb") as f:
            data = f.read()

        if data[:4] != b"MOC3":
            raise ValueError("Not a valid moc3 file")

        def u32(off: int) -> int:
            return struct.unpack_from("<I", data, off)[0]

        def f32(off: int) -> float:
            return struct.unpack_from("<f", data, off)[0]

        # --- CountInfo (section 0) ---
        sec0 = u32(0x40)
        n_verts_total: int = u32(sec0 + 15 * 4)  # index [15] = total vertex count

        # --- CanvasInfo (section 1): ppu, originX, originY, canvasW, canvasH ---
        sec1 = u32(0x44)
        ppu     = f32(sec1 + 0)
        origin_x = f32(sec1 + 4)
        origin_y = f32(sec1 + 8)
        canvas_w = f32(sec1 + 12)
        canvas_h = f32(sec1 + 16)

        if n_verts_total == 0 or canvas_w == 0 or canvas_h == 0:
            raise ValueError("Unexpected zero values in moc3 CountInfo/CanvasInfo")

        # --- Locate the vertex-position section ---
        # It is the first section large enough to hold all vertices (8 bytes each)
        # and whose float values fall within a plausible coordinate range.
        max_coord = max(canvas_w, canvas_h) * 2.5

        offsets: list[int] = []
        for i in range(200):
            off = u32(0x40 + i * 4)
            if off == 0 and i > 5:
                break
            offsets.append(off)

        vert_sec = None
        for i, off in enumerate(offsets):
            size = (offsets[i + 1] - off) if i + 1 < len(offsets) else 0
            if size < n_verts_total * 8:
                continue
            sample = [f32(off + j * 4) for j in range(10)]
            if all(-max_coord < v < max_coord for v in sample):
                vert_sec = off
                break

        if vert_sec is None:
            raise RuntimeError("Could not locate vertex data in moc3")

        # Read default-pose vertex positions (first n_verts_total pairs)
        raw_verts = data[vert_sec: vert_sec + n_verts_total * 8]
        verts = np.frombuffer(raw_verts, dtype="<f4").reshape(n_verts_total, 2)

        # --- Convert pixel coords to canvas fraction (0=top-left, 1=bottom-right) ---
        frac_x = (verts[:, 0] + origin_x) / canvas_w
        frac_y = (verts[:, 1] + origin_y) / canvas_h

        # Keep only in-canvas vertices
        mask = (frac_x > 0.05) & (frac_x < 0.95) & (frac_y > 0.05) & (frac_y < 0.95)
        fx, fy = frac_x[mask], frac_y[mask]
        if len(fy) < 50:
            raise ValueError("Too few in-canvas vertices to detect eye position")

        y_min, y_max = float(fy.min()), float(fy.max())

        # Search for the y-band with highest horizontal spread in the upper face area
        # (two eyes → higher variance in x than forehead/nose/mouth)
        face_cutoff = y_min + (y_max - y_min) * 0.45
        n_bins = 40
        y_edges = np.linspace(y_min, face_cutoff, n_bins + 1)

        best_var, best_cy = -1.0, (y_min + face_cutoff) / 2
        for b in range(n_bins):
            band = (fy >= y_edges[b]) & (fy < y_edges[b + 1])
            if band.sum() < 15:
                continue
            var = float(np.var(fx[band]))
            if var > best_var:
                best_var = var
                best_cy = float((y_edges[b] + y_edges[b + 1]) / 2)

        # Centroid of vertices in the detected eye row
        eye_band = (fy >= best_cy - 0.05) & (fy < best_cy + 0.05)
        if eye_band.sum() >= 5:
            eye_cx = float(fx[eye_band].mean())
            eye_cy = float(fy[eye_band].mean())
        else:
            eye_cx = 0.5
            eye_cy = best_cy

        return round(eye_cy, 3), round(eye_cx, 3)

    def _load_file_content(self, file_path: str) -> str:
        """Load the content of a file with robust encoding handling."""
        # Try common encodings first
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "ascii"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        # If all common encodings fail, try to detect encoding
        try:
            with open(file_path, "rb") as file:
                raw_data = file.read()
            detected = chardet.detect(raw_data)
            detected_encoding = detected["encoding"]

            if detected_encoding:
                try:
                    return raw_data.decode(detected_encoding)
                except UnicodeDecodeError:
                    pass
        except Exception as e:
            logger.error(f"Error detecting encoding for {file_path}: {e}")

        raise UnicodeError(f"Failed to decode {file_path} with any encoding")

    def _lookup_model_info(self, model_name: str) -> dict:
        """
        Find the model information from the model dictionary and return the information about the matched model.

        Parameters:
            model_name (str): The name of the live2d model.

        Returns:
            dict: The dictionary with the information of the matched model.

        Raises:
            FileNotFoundError if the model dictionary file is not found.

            json.JSONDecodeError if the model dictionary file is not a valid JSON file.

            KeyError if the model name is not found in the model dictionary.

        """

        self.live2d_model_name = model_name

        try:
            file_content = self._load_file_content(self.model_dict_path)
            model_dict = json.loads(file_content)
        except FileNotFoundError as file_e:
            logger.critical(
                f"Model dictionary file not found at {self.model_dict_path}."
            )
            raise file_e
        except json.JSONDecodeError as json_e:
            logger.critical(
                f"Error decoding JSON from model dictionary file at {self.model_dict_path}."
            )
            raise json_e
        except UnicodeError as uni_e:
            logger.critical(
                f"Error reading model dictionary file at {self.model_dict_path}."
            )
            raise uni_e
        except Exception as e:
            logger.critical(
                f"Error occurred while reading model dictionary file at {self.model_dict_path}."
            )
            raise e

        # Find the model in the model_dict
        matched_model = next(
            (model for model in model_dict if model["name"] == model_name), None
        )

        if matched_model is None:
            logger.critical(f"Unable to find {model_name} in {self.model_dict_path}.")
            raise KeyError(
                f"{model_name} not found in model dictionary {self.model_dict_path}."
            )

        # The feature: "translate model url to full url if it starts with '/' " is no longer implemented here

        logger.info("Model Information Loaded.")

        return matched_model

    def extract_emotion(self, str_to_check: str) -> list:
        """
        Check the input string for any emotion keywords and return a list of values (the expression index) of the emotions found in the string.

        Parameters:
            str_to_check (str): The string to check for emotions.

        Returns:
            list: A list of values of the emotions found in the string. An empty list is returned if no emotions are found.
        """

        expression_list = []
        str_to_check = str_to_check.lower()

        i = 0
        while i < len(str_to_check):
            if str_to_check[i] != "[":
                i += 1
                continue
            for key in self.emo_map.keys():
                emo_tag = f"[{key}]"
                if str_to_check[i : i + len(emo_tag)] == emo_tag:
                    expression_list.append(self.emo_map[key])
                    i += len(emo_tag) - 1
                    break
            i += 1
        return expression_list

    def extract_gaze(self, text: str) -> Optional[Dict[str, float]]:
        """
        Extract gaze direction from [look_*] or [focus:x,y] tags in the text.

        [focus:x,y] uses normalized screen coordinates (0,0=top-left, 1,1=bottom-right).
        The model's screenPosition in model_dict.json is used as the VTuber's position.
        Returns a dict {"x": -1..1, "y": -1..1} where x>0=right, y>0=up, or None.
        """
        text_lower = text.lower()

        # [focus:x,y] — image-based gaze toward a screen coordinate
        focus_match = re.search(r"\[focus:\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", text_lower)
        if focus_match:
            focus_x = float(focus_match.group(1))
            focus_y = float(focus_match.group(2))
            eye_pos = self._get_eye_screen_position()
            vtuber_x = eye_pos["x"]
            vtuber_y = eye_pos["y"]
            # screen Y increases downward, Live2D Y increases upward → invert
            gaze_x = max(-1.0, min(1.0, (focus_x - vtuber_x) * 2.5))
            gaze_y = max(-1.0, min(1.0, -(focus_y - vtuber_y) * 2.5))
            logger.debug(
                f"[focus] target=({focus_x:.3f},{focus_y:.3f}) "
                f"eye=({vtuber_x:.3f},{vtuber_y:.3f}) "
                f"→ gaze=({gaze_x:.3f},{gaze_y:.3f})"
            )
            return {"x": gaze_x, "y": gaze_y}

        # [look_*] — text-based directional tag; use the last one found
        last_gaze = None
        i = 0
        while i < len(text_lower):
            if text_lower[i] == "[":
                for keyword, gaze in GAZE_KEYWORDS.items():
                    tag = f"[{keyword}]"
                    if text_lower[i : i + len(tag)] == tag:
                        last_gaze = dict(gaze)
                        i += len(tag) - 1
                        break
            i += 1
        return last_gaze

    def extract_movement(self, text: str) -> Optional[Dict[str, float]]:
        """Extract body-movement from [move_*] tags.

        Relative tags (move_left, move_right, …) accumulate their deltas so
        that repeating a tag multiple times compounds the movement.
        [move_center] resets to absolute (0, 0) and clears accumulated delta.

        Returns:
            {"dx": float, "dy": float}   – relative delta to apply
            {"x": 0.0,    "y": 0.0}      – absolute reset (move_center)
            None if no movement tag found.
        """
        text_lower = text.lower()
        total_dx = 0.0
        total_dy = 0.0
        found = False
        is_abs_reset = False

        i = 0
        while i < len(text_lower):
            if text_lower[i] == "[":
                for keyword, delta in MOVEMENT_KEYWORDS.items():
                    tag = f"[{keyword}]"
                    if text_lower[i : i + len(tag)] == tag:
                        if delta.get("abs"):
                            # move_center: absolute reset, discard accumulated delta
                            is_abs_reset = True
                            total_dx = 0.0
                            total_dy = 0.0
                        else:
                            total_dx += delta["dx"]
                            total_dy += delta["dy"]
                        found = True
                        i += len(tag) - 1
                        break
            i += 1

        if not found:
            return None
        if is_abs_reset:
            return {"x": 0.0, "y": 0.0}   # absolute: go to center
        return {"dx": total_dx, "dy": total_dy}  # relative delta

    def extract_model_scale(self, text: str) -> Optional[float]:
        """Extract scale multiplier from [scale_*] tags. Returns last found."""
        text_lower = text.lower()
        last_scale = None
        i = 0
        while i < len(text_lower):
            if text_lower[i] == "[":
                for keyword, multiplier in SCALE_KEYWORDS.items():
                    tag = f"[{keyword}]"
                    if text_lower[i : i + len(tag)] == tag:
                        last_scale = multiplier
                        i += len(tag) - 1
                        break
            i += 1
        return last_scale

    def remove_gaze_keywords(self, text: str) -> str:
        """Remove gaze direction tags ([look_*] and [focus:x,y]) from text."""
        text = re.sub(r"\[focus:\s*[0-9.]+\s*,\s*[0-9.]+\s*\]", "", text, flags=re.IGNORECASE)
        for keyword in GAZE_KEYWORDS:
            text = re.sub(re.escape(f"[{keyword}]"), "", text, flags=re.IGNORECASE)
        return text

    def remove_emotion_keywords(self, target_str: str) -> str:
        """
        Remove the emotion keywords from the input string and return the cleaned string.

        Parameters:
            str_to_check (str): The string to check for emotions.

        Returns:
            str: The cleaned string with the emotion keywords removed.
        """

        lower_str = target_str.lower()

        for key in self.emo_map.keys():
            lower_key = f"[{key}]".lower()
            while lower_key in lower_str:
                start_index = lower_str.find(lower_key)
                end_index = start_index + len(lower_key)
                target_str = target_str[:start_index] + target_str[end_index:]
                lower_str = lower_str[:start_index] + lower_str[end_index:]
        return target_str

    def extract_volume_adjustment(self, text: str) -> Optional[int]:
        """Extract cumulative volume adjustment from [volume_up] / [volume_down] tags.

        Each tag adds ±5 to the total. Multiple tags in the same sentence accumulate.
        Returns the total integer adjustment, or None if no volume tag is found.
        """
        text_lower = text.lower()
        total = 0
        found = False
        i = 0
        while i < len(text_lower):
            if text_lower[i] == "[":
                for keyword, delta in VOLUME_KEYWORDS.items():
                    tag = f"[{keyword}]"
                    if text_lower[i : i + len(tag)] == tag:
                        total += delta
                        found = True
                        i += len(tag) - 1
                        break
            i += 1
        return total if found else None

    def remove_volume_keywords(self, text: str) -> str:
        """Remove [volume_up] and [volume_down] tags from text."""
        for keyword in VOLUME_KEYWORDS:
            text = re.sub(re.escape(f"[{keyword}]"), "", text, flags=re.IGNORECASE)
        return text
