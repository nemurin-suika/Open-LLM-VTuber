import json
import re
import chardet
from typing import Dict, Optional
from loguru import logger

# Gaze direction presets: x = left(-1) to right(+1), y = down(-1) to up(+1)
GAZE_KEYWORDS: Dict[str, Dict[str, float]] = {
    "look_left": {"x": -0.8, "y": 0.0},
    "look_right": {"x": 0.8, "y": 0.0},
    "look_up": {"x": 0.0, "y": 0.6},
    "look_down": {"x": 0.0, "y": -0.5},
    "look_center": {"x": 0.0, "y": 0.0},
    "look_upleft": {"x": -0.6, "y": 0.5},
    "look_upright": {"x": 0.6, "y": 0.5},
    "look_downleft": {"x": -0.6, "y": -0.4},
    "look_downright": {"x": 0.6, "y": -0.4},
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
        focus_match = re.search(r"\[focus:([0-9.]+),([0-9.]+)\]", text_lower)
        if focus_match:
            focus_x = float(focus_match.group(1))
            focus_y = float(focus_match.group(2))
            screen_pos = self.model_info.get("screenPosition", {"x": 0.85, "y": 0.85})
            vtuber_x = float(screen_pos.get("x", 0.85))
            vtuber_y = float(screen_pos.get("y", 0.85))
            # screen Y increases downward, Live2D Y increases upward → invert
            gaze_x = max(-1.0, min(1.0, (focus_x - vtuber_x) * 2.5))
            gaze_y = max(-1.0, min(1.0, -(focus_y - vtuber_y) * 2.5))
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

    def remove_gaze_keywords(self, text: str) -> str:
        """Remove gaze direction tags ([look_*] and [focus:x,y]) from text."""
        text = re.sub(r"\[focus:[0-9.]+,[0-9.]+\]", "", text, flags=re.IGNORECASE)
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
