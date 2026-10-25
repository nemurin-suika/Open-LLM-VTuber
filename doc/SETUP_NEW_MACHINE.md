# 새 맥에 설치하기

이 문서는 아무것도 없는 맥에 이 스택(백엔드 + letta + 프론트엔드 + 목소리)을
올려서 대화가 되는 상태까지 만드는 절차다. 에이전트(Claude Code, Codex 등)가
읽고 그대로 수행할 수 있게 썼다.

구성 범위는 **대화 + TTS + Live2D + DB** 다. 마이크 입력(ASR), 시스템 오디오
분석, 라이브 플랫폼 연동은 포함하지 않는다. 필요해지면 나중에 켜면 된다.

---

## 0. 요약 (세 줄)

```sh
git clone https://github.com/nemurin-suika/Open-LLM-VTuber.git \
    -b v1-release-nemurin ~/develop/Open-LLM-VTuber-v1.2.1-en
cd ~/develop/Open-LLM-VTuber-v1.2.1-en && ./scripts/bootstrap.sh
./scripts/start-all.sh
```

그 뒤 손으로 할 일은 네 가지다. **Live2D 모델 폴더 복사**,
**목소리 참조 음성 복사**, **페르소나 작성**, **claude CLI 로그인**.
자세한 건 아래.

`bootstrap.sh` 는 GPT-SoVITS 모델까지 받으므로 처음 한 번은 삼십 분 이상
걸린다고 보면 된다. 목소리를 빼고 먼저 대화만 띄워 보려면
`./scripts/bootstrap.sh --skip-tts` 로 돌리고, `conf.yaml` 의 `tts_model` 을
`edge_tts` 로 두면 된다.

---

## 1. 저장소 구조

네 저장소가 한 상위 디렉토리 아래 나란히 있어야 한다.

```
~/develop/
├── Open-LLM-VTuber-v1.2.1-en/      백엔드 (이 저장소)
│   └── frontend -> ../Open-LLM-VTuber-Web   (심볼릭 링크)
├── Open-LLM-VTuber-Web/            프론트엔드
├── letta/                          기억·에이전트 서버
└── GPT-SoVITS-v3lora-20250228/     목소리 (TTS 추론 서버)
```

| 저장소 | 브랜치 | 비고 |
| --- | --- | --- |
| `nemurin-suika/Open-LLM-VTuber` | `v1-release-nemurin` | 디렉토리 이름을 `Open-LLM-VTuber-v1.2.1-en` 으로 둔다 |
| `nemurin-suika/Open-LLM-VTuber-Web` | `nemurin` | 백엔드의 `frontend` 가 이걸 가리킨다 |
| `nemurin-suika/letta` | `nemurin` | |
| `nemurin-suika/GPT-SoVITS` | `nemurin` | 디렉토리 이름을 `GPT-SoVITS-v3lora-20250228` 으로 둔다 |

GPT-SoVITS 는 파이썬 3.9 전용 환경을 저장소 안의 `.venv` 에 따로 둔다.
백엔드·letta 는 `uv` 로 각자 환경을 쓰므로 서로 간섭하지 않는다.
목소리를 안 쓸 거면 이 저장소는 없어도 된다(`--skip-tts`).

`frontend` 는 디렉토리가 아니라 **심볼릭 링크**다. 백엔드 `run.sh` 가
`pushd frontend && npm run build:web` 을 하고, 서버는 `frontend/dist/web` 을
정적 서빙한다. 링크가 끊겨 있으면 빈 화면이 뜬다.

---

## 2. bootstrap.sh 가 해 주는 것

```sh
./scripts/bootstrap.sh              # 대화형
./scripts/bootstrap.sh --yes        # 되묻지 않고 전부 진행
./scripts/bootstrap.sh --dir ~/work # 저장소를 다른 상위 디렉토리에
./scripts/bootstrap.sh --skip-brew --skip-npm
./scripts/bootstrap.sh --skip-tts   # 목소리는 나중에
```

1. 필요한 도구 점검 — `postgresql@17`, `redis`, `node`, `ollama`, `uv`, `claude`
2. letta / 프론트엔드 / GPT-SoVITS 저장소 클론
3. `frontend` 심볼릭 링크 연결
4. `letta/run.sh` 의 절대경로·LAN 주소를 이 맥 기준으로 치환
5. `~/.secrets` 템플릿 생성 (없을 때만)
6. PostgreSQL 데이터 디렉토리 `initdb` (`~/.mizuki/pgdata`)
7. `uv sync` (백엔드·letta), `npm install` (프론트엔드)
8. GPT-SoVITS 설치 — `GPT-SoVITS/scripts/setup-mac.sh` 를 불러서 수행
9. `ollama pull bge-m3`
10. 런타임 디렉토리 생성 (`cache`, `live2d-models`, `avatars` …)
11. `conf.yaml` 생성

**하지 않는 것**: launchd 등록, Live2D 모델 설치, TTS 참조 음성 설치, API 키 입력.

### 8번 GPT-SoVITS 설치가 하는 일

`setup-mac.sh` 는 한국어 추론 경로만 깐다. 그래서 받는 양이 적다.

- `brew install python@3.9 ffmpeg` 점검 — `requirements.txt` 가
  `numpy==1.23.4`, `numba==0.56.4` 로 고정이라 3.10 이상에서는 휠이 없어 깨진다.
  `ffmpeg` 는 CLI 바이너리로 호출되므로 반드시 있어야 한다.
- 저장소 안에 `.venv` 생성 → `requirements.txt` 설치 (+ 빠지기 쉬운 `hangulize`)
- NLTK 데이터 — 한국어 문장에 섞인 영어 단어가 품사 태거를 지나간다.
  없으면 첫 추론에서 `LookupError` 로 멈춘다.
- 사전학습 모델 약 1.3G — `chinese-hubert-base`,
  `chinese-roberta-wwm-ext-large`, `gsv-v2final-pretrained` 의 `s1`/`s2G`,
  `fast_langdetect/lid.176.bin`. 이어받기가 되므로 끊기면 다시 돌리면 된다.

받지 않는 것: UVR5, ASR, G2PW, BigVGAN, v3 가중치. 전부 학습용이거나
중국어·v3 전용이고 합치면 수 기가다. `api_v2` 는 v2 로만 돌고,
**한국어는 v2 에서만 지원**하므로 v3 쪽은 아예 쓰지 않는다.

### 치환에 대한 주의

4번에서 `letta/run.sh` 를 직접 고친다. 그 저장소에 로컬 수정이 남으므로
**그 변경은 커밋하지 않는다**. 되돌리려면 `git -C <letta> checkout run.sh`.

---

## 3. conf.yaml

`conf.yaml` 은 페르소나 프롬프트를 담고 있어 **저장소에 올라가지 않는다**
(`.gitignore` 대상). 그래서 설치할 때마다 템플릿에서 생성한다.

```sh
python3 scripts/make_conf.py                 # 대화형으로 물어보며 생성
python3 scripts/make_conf.py --defaults      # 전부 기본값
python3 scripts/make_conf.py --persona ~/persona.txt --force
python3 scripts/make_conf.py --set port=12394 --set tts_model=gpt_sovits_tts
```

- 템플릿: `config_templates/conf.work.template.yaml` (여기엔 프롬프트도 키도 없다)
- 이미 `conf.yaml` 이 있으면 `--force` 없이는 덮어쓰지 않고, 덮어쓸 때는
  `conf.yaml.bak-<시각>` 으로 백업한다.
- 생성 후 PyYAML 이 있으면 문법과 핵심 키를 검사한다.

### 지정할 수 있는 항목

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `host` | `127.0.0.1` | 다른 기기에서 접속하려면 `0.0.0.0` |
| `port` | `12393` | 백엔드 포트 |
| `enable_screen_capture` | `false` | 화면 캡처를 LLM 에 넘길지 |
| `conf_name` / `conf_uid` | `default` / `default_001` | 설정 이름·식별자 |
| `character_name` / `human_name` | `미즈키` / `사용자` | 표시 이름 |
| `live2d_model_name` | `MIA` | `model_dict.json` 의 `name` 과 같아야 함 |
| `letta_host` / `letta_port` | `localhost` / `8283` | letta 서버 |
| `letta_model` | `claude_cli/claude-opus-5` | letta 쪽 LLM handle |
| `letta_embedding` | `ollama/bge-m3:latest` | 임베딩 handle |
| `context_window_limit` | `100000` | 컨텍스트 상한 |
| `asr_model` | `null` | 마이크를 쓰면 `sherpa_onnx_asr` |
| `vad_model` | `null` | 마이크를 쓰면 `silero_vad` |
| `tts_model` | `gpt_sovits_tts` | 설치 없이 소리만 보려면 `edge_tts` |
| `edge_tts_voice` | `ko-KR-SunHiNeural` | `ko-KR-SeoHyeonNeural` 등 |
| `gpt_sovits_api_url` | `http://127.0.0.1:9880/tts` | 같은 기기에 함께 띄운다 |
| `gpt_sovits_ref_audio` | `saki.wav` | GPT-SoVITS 저장소 루트 기준 상대경로 |
| `gpt_sovits_prompt_text` | 참조 음성의 대사 | 음성 내용과 글자가 맞아야 발음이 산다 |

### letta 에이전트 id 는 비워 둔다

`agent_config.agent_settings.letta_agent.id` 는 빈 문자열로 생성된다.
백엔드 `run.sh` 가 기동할 때 `Open-LLM-VTuber` 라는 이름의 에이전트를 찾고,
없으면 만들고, 그 id 를 **conf.yaml 에 자동으로 적어 넣는다**. 손으로 채울
필요가 없다. 같은 블록에서 시스템 프롬프트·툴·메모리 블록도 맞춰 준다.

### ASR 을 끄는 것에 대해

`asr_model: null` 이면 ASR 엔진을 아예 초기화하지 않는다. sherpa-onnx 모델
파일(수백 MB)을 받아 둘 필요가 없다. 이 상태에서 음성 입력이 들어오면
에러 메시지를 돌려주고 대화는 계속된다. 마이크를 쓰려면 `asr_model` 과
`vad_model` 을 함께 채우고 모델 파일을 받아야 한다.

---

## 4. 띄우기

```sh
./scripts/start-all.sh           # letta → 목소리 → 백엔드 순서로 기동
./scripts/start-all.sh --build   # 프론트엔드를 다시 빌드하고 띄운다
./scripts/start-all.sh --no-tts  # 목소리 서버는 띄우지 않는다
./scripts/start-all.sh --status  # 상태만
./scripts/start-all.sh --stop    # 전부 정지
```

브라우저에서 `http://127.0.0.1:12393`.

**순서가 중요하다.** 백엔드 `run.sh` 안의 letta 에이전트 설정 블록은 8283 에
한 번만 접속을 시도하고, 실패하면 조용히 건너뛴 채로 서버를 띄운다. 그러면
시스템 프롬프트와 메모리 블록 갱신이 누락된 상태로 돌아간다.
`start-all.sh` 는 letta health 가 `ok` 가 될 때까지 기다린 뒤 백엔드를 띄운다.

목소리(9880)는 `conf.yaml` 의 `tts_model` 이 `gpt_sovits_tts` 일 때만 띄운다.
참조 음성과 대사는 `conf.yaml` 값을 그대로 환경 변수로 넘기므로 설정 한 곳만
고치면 양쪽이 같이 따라온다. 모델을 올리는 데 시간이 걸려서 백엔드보다 먼저
띄우고 포트가 열릴 때까지 기다리지만, 늦게 떠도 대화 자체는 된다(그 몇 마디만
소리 없이 지나간다). 로그는 `~/.mizuki/logs/gpt-sovits.log`.

첫 기동에서 `frontend/dist/web` 이 없으면 자동으로 빌드하고 띄운다.

상주 운영(로그인하면 자동 기동)이 필요하면 launchd 로 따로 등록한다.
이 스크립트는 터미널에서 직접 돌리는 용도다.

---

## 5. 손으로 옮겨야 하는 것

### 5-1. Live2D 모델

라이선스 때문에 저장소에 들어 있지 않다. 모델 폴더를 통째로
`live2d-models/<모델이름>/` 에 넣고 `model_dict.json` 에 항목을 추가한다.

붙이는 절차, 필수 필드, VTube Studio 배포 모델의 함정(표정이 등록돼 있지
않다거나 `LipSync.Ids` 가 비어 있다거나)은 **`doc/LIVE2D_MODEL_SETUP.md`** 에
전부 정리돼 있다. 모델을 새로 붙일 때는 그 문서를 먼저 읽는다.

`conf.yaml` 의 `live2d_model_name` 과 `model_dict.json` 의 `name` 이 같아야
한다. 다르면 모델이 로드되지 않는다.

### 5-2. 목소리 참조 음성

참조 음성은 저작권 자료라 저장소에 들어 있지 않다(`.gitignore` 대상).
파일 하나를 GPT-SoVITS 저장소 **루트**에 둔다.

```sh
# mp3 는 읽지 못한다. 먼저 변환한다.
ffmpeg -i saki.mp3 -c:a pcm_s16le ~/develop/GPT-SoVITS-v3lora-20250228/saki.wav
```

조건은 세 가지다.

1. **wav(PCM)**. mp3 를 그대로 주면 참조 음성을 읽는 단계에서 실패한다.
2. **3 초에서 10 초**. 배경음·효과음·다른 목소리가 섞이지 않은 토막이 좋다.
3. **대사가 설정과 일치**. `gpt_sovits_prompt_text` 에 그 음성이 실제로 말하는
   문장을 그대로 적는다. 글자가 어긋나면 발음이 뭉개진다.
   이 값은 기본값이 비어 있다. **사람이 직접 들어보고 받아 적는다.**
   자동 전사(ASR)는 조사나 어미를 틀리게 잡는 일이 있어서 기본값으로 두지
   않는다.

학습한 화자 가중치는 쓰지 않는다. 사전학습 v2 모델에 참조 음성만 물리는
방식이라 wav 한 개를 바꾸면 목소리가 바뀐다.

### 5-3. 로그인과 키

- `claude auth login` — letta 가 `claude_cli` 모델을 쓸 때 필요하다.
- `~/.secrets` — `letta/run.sh` 가 첫 줄에서 읽는다. 없으면 기동이 멈춘다.
  bootstrap 이 빈 템플릿을 만들어 두므로 필요한 키만 채운다.

### 5-4. 페르소나

`conf.yaml` 의 `persona_prompt` 가 자리표시 문구로 들어간다. 직접 고치거나,
파일로 준비해서 다시 생성한다.

```sh
python3 scripts/make_conf.py --persona ~/persona.txt --force
```

---

## 6. 안 되는 것들 점검표

| 증상 | 먼저 볼 곳 |
| --- | --- |
| 브라우저가 빈 화면 | `frontend` 링크, `frontend/dist/web` 존재 여부 → `--build` 로 재기동 |
| 캐릭터가 안 뜸 | `conf.yaml` 의 `live2d_model_name` 과 `model_dict.json` 의 `name` 일치 여부 |
| 표정이 안 바뀜 | `model_dict.json` 의 `emotionMap`, 모델의 `Expressions` 등록 여부 |
| 입이 안 움직임 | model3.json 의 `Groups` 에 `LipSync` 가 있고 `Ids` 가 비어 있지 않은지 |
| 말투·기억이 예전 그대로 | letta 보다 백엔드가 먼저 떴을 때 생긴다. `--stop` 후 `start-all.sh` 로 다시 |
| letta 가 안 뜸 | `tail -n 100 ~/.mizuki/logs/letta.log`, postgres/redis 기동 여부 |
| postgres 가 바로 죽음 | 로케일. `LANG`/`LC_ALL` 이 `en_US.UTF-8` 인지 |
| 포트 점유 | `./scripts/start-all.sh --status` → `--stop` |
| 소리가 안 남 | `--status` 로 9880 확인 → `tail -n 100 ~/.mizuki/logs/gpt-sovits.log` |
| 목소리 서버가 바로 죽음 | `tts_infer_mac.yaml` 을 쓰는지. 기본 `tts_infer.yaml` 은 `device: cuda` 라 맥에서 터진다 |
| 참조 음성을 못 찾음 | `conf.yaml` 의 `ref_audio_path` 가 GPT-SoVITS 저장소 루트 기준 상대경로인지 |
| 발음이 뭉개짐 | `prompt_text` 가 참조 음성의 실제 대사와 같은지 |
| 첫 추론에서 `LookupError` | NLTK 데이터 누락. `scripts/setup-mac.sh --skip-models --skip-deps` 로는 안 고쳐지니 설치 단계를 다시 |

---

## 7. 집 운영 환경과 다른 점

참고용. 이 문서대로 깐 환경은 아래가 빠져 있다.

- launchd 상주 등록, 자동 재기동 오케스트레이션
- 주기 백업(pg_dump), WAL 아카이빙
- 시스템 오디오 분석 서버 연동
- 라이브 플랫폼(치지직 등) 연동
- 마이크 입력

목소리 구성도 다르다. 집에서는 TTS 를 데스크톱 GPU 쪽에 두고 LAN 으로
불러 쓰며, 그쪽이 꺼져 있을 때 같은 기기의 서버로 넘어가게 해 두었다.
여기서는 폴백 없이 같은 기기 하나만 쓴다. GPU 가 없으므로 추론은 `mps` 에서
돌고, 첫 한 마디는 모델을 올리는 시간만큼 늦게 나온다.

필요해지면 각각 따로 올리면 된다. 이 문서의 범위는 "대화가 되는 상태"까지다.
