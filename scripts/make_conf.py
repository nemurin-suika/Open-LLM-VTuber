#!/usr/bin/env python3
"""conf.yaml 생성기.

config_templates/conf.work.template.yaml 의 __PLACEHOLDER__ 를 채워서
레포 루트에 conf.yaml 을 만든다.

conf.yaml 에는 페르소나 프롬프트가 들어가므로 저장소에 커밋하지 않는다
(.gitignore 에 이미 등록돼 있다). 그래서 설치할 때마다 이 스크립트로 새로
만들어 쓰는 구조다.

사용법:
    python3 scripts/make_conf.py                      # 대화형
    python3 scripts/make_conf.py --defaults           # 전부 기본값
    python3 scripts/make_conf.py --persona persona.txt --defaults
    python3 scripts/make_conf.py --set port=12394 --set tts_model=gpt_sovits_tts

표준 라이브러리만 쓴다(검증 단계의 PyYAML 은 있으면 쓰고 없으면 건너뛴다).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "config_templates" / "conf.work.template.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "conf.yaml"

PLACEHOLDER_PERSONA = "__PERSONA_PROMPT__"

DEFAULT_PERSONA = """너는 사용자의 데스크톱에서 함께 일하는 AI 캐릭터다.
말투는 편하게, 답은 짧고 분명하게 한다.
소리내어 읽힐 글이므로 마크다운 기호나 목록 기호는 쓰지 않는다.

여기는 자리표시 문구다. 실제 페르소나로 바꿔 쓸 것.
"""


class Field:
    """치환 항목 하나.

    quote=True  이면 YAML 에 작은따옴표로 감싸 들어가는 값(템플릿 쪽에
                이미 따옴표가 있으므로 값만 넣는다).
    raw=True    이면 null / 숫자 / True 처럼 따옴표 없이 들어가는 값.
    """

    def __init__(self, key, placeholder, default, prompt, choices=None, raw=False):
        self.key = key
        self.placeholder = placeholder
        self.default = default
        self.prompt = prompt
        self.choices = choices
        self.raw = raw


FIELDS = [
    Field("host", "__HOST__", "127.0.0.1",
          "백엔드 바인딩 주소 (다른 기기에서 접속하려면 0.0.0.0)"),
    Field("port", "__PORT__", "12393", "백엔드 포트", raw=True),
    Field("enable_screen_capture", "__ENABLE_SCREEN_CAPTURE__", "false",
          "화면 캡처를 LLM 에 넘길까", choices=["true", "false"], raw=True),

    Field("conf_name", "__CONF_NAME__", "default", "설정 이름"),
    Field("conf_uid", "__CONF_UID__", "default_001", "설정 고유 id"),
    Field("character_name", "__CHARACTER_NAME__", "미즈키", "캐릭터 표시 이름"),
    Field("human_name", "__HUMAN_NAME__", "사용자", "사용자 표시 이름"),
    Field("live2d_model_name", "__LIVE2D_MODEL_NAME__", "MIA",
          "Live2D 모델 이름 (model_dict.json 의 name 과 같아야 함)"),

    Field("letta_host", "__LETTA_HOST__", "localhost", "letta 서버 주소"),
    Field("letta_port", "__LETTA_PORT__", "8283", "letta 서버 포트", raw=True),
    Field("letta_model", "__LETTA_MODEL__", "claude_cli/claude-opus-5",
          "letta LLM handle"),
    Field("letta_embedding", "__LETTA_EMBEDDING__", "ollama/bge-m3:latest",
          "letta 임베딩 handle"),
    Field("context_window_limit", "__CONTEXT_WINDOW_LIMIT__", "100000",
          "컨텍스트 윈도 상한", raw=True),

    Field("asr_model", "__ASR_MODEL__", "null",
          "음성 인식 (마이크 안 쓰면 null)",
          choices=["null", "sherpa_onnx_asr"], raw=True),
    Field("vad_model", "__VAD_MODEL__", "null",
          "음성 활동 감지 (마이크 안 쓰면 null)",
          choices=["null", "silero_vad"], raw=True),

    Field("tts_model", "__TTS_MODEL__", "gpt_sovits_tts",
          "TTS 엔진", choices=["edge_tts", "gpt_sovits_tts"]),
    Field("edge_tts_voice", "__EDGE_TTS_VOICE__", "ko-KR-SunHiNeural",
          "edge_tts 음성 이름 (gpt_sovits 를 쓸 수 없을 때의 대체용)"),
    Field("gpt_sovits_api_url", "__GPT_SOVITS_API_URL__",
          "http://127.0.0.1:9880/tts", "GPT-SoVITS API 주소"),
    Field("gpt_sovits_ref_audio", "__GPT_SOVITS_REF_AUDIO__", "saki.wav",
          "GPT-SoVITS 참조 음성 파일 (GPT-SoVITS 루트 기준 상대경로)"),
    Field("gpt_sovits_prompt_text", "__GPT_SOVITS_PROMPT_TEXT__", "",
          "참조 음성에서 말하는 내용 그대로 (사람이 직접 받아 적는다)"),
]

# tts_model 이 edge_tts 일 때는 물어볼 필요가 없는 항목들.
GPT_SOVITS_ONLY = {
    "gpt_sovits_api_url", "gpt_sovits_ref_audio", "gpt_sovits_prompt_text",
}
EDGE_ONLY = {"edge_tts_voice"}


def ask(field: Field) -> str:
    suffix = f" [{field.default}]" if field.default != "" else " []"
    if field.choices:
        suffix = f" ({'/'.join(field.choices)})" + suffix
    while True:
        try:
            answer = input(f"  {field.prompt}{suffix}: ").strip()
        except EOFError:
            print()
            return field.default
        if answer == "":
            return field.default
        if field.choices and answer not in field.choices:
            print(f"    → {', '.join(field.choices)} 중에서 고르세요.")
            continue
        return answer


def indent_block(text: str, spaces: int = 4) -> str:
    """YAML 블록 스칼라 본문으로 쓸 수 있게 들여쓴다."""
    pad = " " * spaces
    lines = text.rstrip("\n").split("\n")
    return "\n".join(pad + line if line.strip() else "" for line in lines)


def validate(path: Path) -> bool:
    """생성 결과가 YAML 로 읽히는지, 핵심 키가 들어있는지 본다."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        print("  (PyYAML 이 없어 문법 검증은 건너뜁니다)")
        return True

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: YAML 파싱 실패 — {exc}")
        return False

    char = (data or {}).get("character_config") or {}
    missing = [k for k in ("conf_name", "live2d_model_name", "agent_config",
                           "tts_config") if k not in char]
    if missing:
        print(f"  ERROR: character_config 에 누락된 키 — {', '.join(missing)}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="conf.yaml 생성기")
    parser.add_argument("--defaults", action="store_true",
                        help="질문 없이 기본값으로 생성")
    parser.add_argument("--persona", metavar="FILE",
                        help="페르소나 프롬프트를 담은 텍스트 파일")
    parser.add_argument("--output", metavar="FILE", default=str(DEFAULT_OUTPUT),
                        help=f"생성 위치 (기본 {DEFAULT_OUTPUT})")
    parser.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                        help="항목을 직접 지정 (여러 번 쓸 수 있음)")
    parser.add_argument("--force", action="store_true",
                        help="이미 있어도 덮어쓴다 (기존 파일은 백업)")
    args = parser.parse_args()

    if not TEMPLATE.exists():
        print(f"ERROR: 템플릿이 없습니다 — {TEMPLATE}", file=sys.stderr)
        return 1

    output = Path(args.output).expanduser().resolve()

    overrides = {}
    for item in args.set:
        if "=" not in item:
            print(f"ERROR: --set 은 KEY=VALUE 형식이어야 합니다 — {item}",
                  file=sys.stderr)
            return 1
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()

    known = {f.key for f in FIELDS}
    unknown = set(overrides) - known
    if unknown:
        print(f"ERROR: 모르는 항목 — {', '.join(sorted(unknown))}", file=sys.stderr)
        print(f"  쓸 수 있는 항목: {', '.join(sorted(known))}", file=sys.stderr)
        return 1

    if output.exists() and not args.force:
        print(f"ERROR: 이미 있습니다 — {output}", file=sys.stderr)
        print("  덮어쓰려면 --force 를 붙이세요 (기존 파일은 백업됩니다).",
              file=sys.stderr)
        return 1

    interactive = not args.defaults and sys.stdin.isatty()
    if interactive:
        print("conf.yaml 을 만듭니다. 그냥 Enter 를 누르면 대괄호 안 기본값이 들어갑니다.\n")

    values = {}
    for field in FIELDS:
        if field.key in overrides:
            values[field.key] = overrides[field.key]
            continue
        if not interactive:
            values[field.key] = field.default
            continue
        # TTS 엔진에 따라 필요 없는 질문은 건너뛴다.
        tts = values.get("tts_model", "edge_tts")
        if field.key in GPT_SOVITS_ONLY and tts != "gpt_sovits_tts":
            values[field.key] = field.default
            continue
        if field.key in EDGE_ONLY and tts != "edge_tts":
            values[field.key] = field.default
            continue
        values[field.key] = ask(field)

    # 페르소나
    if args.persona:
        persona_path = Path(args.persona).expanduser()
        if not persona_path.exists():
            print(f"ERROR: 페르소나 파일이 없습니다 — {persona_path}", file=sys.stderr)
            return 1
        persona = persona_path.read_text(encoding="utf-8")
    else:
        persona = DEFAULT_PERSONA

    content = TEMPLATE.read_text(encoding="utf-8")
    for field in FIELDS:
        content = content.replace(field.placeholder, values[field.key])
    content = content.replace(PLACEHOLDER_PERSONA, indent_block(persona))

    all_placeholders = [f.placeholder for f in FIELDS] + [PLACEHOLDER_PERSONA]
    leftovers = [p for p in all_placeholders if p in content]
    if leftovers:
        print(f"ERROR: 치환되지 않은 자리가 남았습니다 — {', '.join(leftovers)}",
              file=sys.stderr)
        return 1

    header = (
        "# 이 파일은 scripts/make_conf.py 가 "
        f"config_templates/conf.work.template.yaml 에서 생성했습니다.\n"
        f"# 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "# 페르소나 프롬프트가 들어 있으므로 저장소에 커밋하지 마세요.\n"
        "\n"
    )
    content = header + content

    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = output.with_name(f"{output.name}.bak-{stamp}")
        shutil.copy2(output, backup)
        print(f"\n기존 파일 백업: {backup}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    os.chmod(output, 0o600)

    print(f"\n생성 완료: {output}")
    if not validate(output):
        return 1

    print("\n확인할 것:")
    print(f"  - live2d_model_name 은 '{values['live2d_model_name']}' 입니다. "
          "model_dict.json 에 같은 이름이 있어야 합니다.")
    if args.persona:
        print(f"  - 페르소나는 {args.persona} 에서 가져왔습니다.")
    else:
        print("  - 페르소나가 자리표시 문구입니다. persona_prompt 를 직접 채우세요.")
    if values["tts_model"] == "gpt_sovits_tts":
        print(f"  - GPT-SoVITS 서버({values['gpt_sovits_api_url']})가 떠 있어야 하고, "
              f"참조 음성 '{values['gpt_sovits_ref_audio']}' 를 옮겨 둬야 합니다.")
        if values["gpt_sovits_prompt_text"].strip() == "":
            print("  - prompt_text 가 비어 있습니다. 참조 음성이 말하는 내용을 "
                  "직접 받아 적어 conf.yaml 의 prompt_text 에 넣으세요.")
            print("    (자동 전사는 틀릴 수 있어서 기본값을 두지 않습니다. "
                  "대사가 어긋나면 발음이 뭉개집니다.)")
    if values["asr_model"] == "null":
        print("  - 마이크 입력은 꺼져 있습니다. 쓰려면 asr_model 과 vad_model 을 채우세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
