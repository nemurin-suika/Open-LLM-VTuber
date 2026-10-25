# Live2D 모델 연결 가이드

새 Live2D 모델을 이 프로젝트에 붙일 때 필요한 설정을 처음부터 끝까지 정리한 문서다.
**사람이 읽어도 되고, Claude Code / Codex 같은 코딩 에이전트에게 이 파일 하나만 주고 "이 가이드대로 모델 붙여줘"라고 시켜도 되도록** 작성했다.

- 백엔드 레포 루트를 `BE`, 프론트 레포(`Open-LLM-VTuber-Web`) 루트를 `FE`로 표기한다.
- 모든 사실에는 근거 파일·라인을 달았다. 코드가 바뀌면 라인은 어긋날 수 있으니 파일명을 기준으로 다시 찾을 것.

---

## 0. 에이전트용 요약 (이것만 지키면 동작한다)

1. 모델 런타임 일체를 `BE/live2d-models/<폴더>/` 아래에 둔다.
2. `<폴더>` 안의 `*.model3.json`을 열어 `FileReferences.Expressions` 배열의 **순서(0-based 인덱스)** 와 각 항목의 `Name`을 적어둔다.
3. `BE/model_dict.json` (최상위가 **배열**) 에 엔트리를 하나 추가한다. `name`, `url`, `kScale`, `initialXshift`, `initialYshift`, `emotionMap` 은 반드시 넣는다.
4. `emotionMap` 의 키는 **LLM이 대사에 쓸 감정 키워드**, 값은 **Expressions 인덱스(숫자) 또는 Name(문자열)** 이다.
5. `BE/conf.yaml` 의 `character_config.live2d_model_name` 을 3번의 `name` 과 **문자 단위로 동일하게** 맞춘다.
6. 백엔드를 재시작한다. `model_dict.json` 은 기동 시 1회만 읽힌다.
7. 프론트에 옛 모델이 계속 보이면 브라우저/Electron **localStorage 의 `modelInfo` 키**를 지운다.

---

## 1. 전체 데이터 흐름

```
model_dict.json ──(BE 기동 시 1회 로드)──> Live2dModel
      │                                        │
      │                             emotionMap 키 → 시스템 프롬프트 주입
      │                                        │
      └──(dict 통째로 WebSocket 전송)──> FE ModelInfo ──> Cubism WebSDK
                 type: "set-model-and-conf"
```

- 로드·매칭: `BE/src/open_llm_vtuber/live2d_model.py` (`__init__` 및 모델 조회부)
- 전송: `BE/src/open_llm_vtuber/websocket_handler.py` 의 `set-model-and-conf` 메시지, 키 이름은 `model_info`
- 수신: `FE/src/renderer/src/services/websocket-handler.tsx` — 상대 URL이면 `baseUrl` 을 붙여 절대화
- 저장: `FE/src/renderer/src/context/live2d-config-context.tsx` 의 `setModelInfo`
- 적용: `FE/src/renderer/src/hooks/canvas/use-live2d-model.ts`

> 이 프로젝트는 `pixi-live2d-display` 를 **쓰지 않는다.** 번들된 공식 Cubism Web SDK (`FE/src/renderer/WebSDK/`) 의 `LAppModel` / `LAppAdapter` 를 직접 호출한다. 인터넷에서 찾은 pixi 기반 예제를 그대로 적용하지 말 것.

---

## 2. 파일 배치 규칙

`BE/live2d-models/` 디렉토리가 `/live2d-models` URL 로 정적 마운트된다 (`BE/src/open_llm_vtuber/server.py`).

허용되는 배치 예:

```
BE/live2d-models/mymodel/mymodel.model3.json          → url: /live2d-models/mymodel/mymodel.model3.json
BE/live2d-models/mymodel/runtime/mymodel.model3.json  → url: /live2d-models/mymodel/runtime/mymodel.model3.json
```

**제약**: 프론트 `parseModelUrl` (`FE/src/renderer/src/hooks/canvas/use-live2d-model.ts`) 이 URL 의 **끝에서 두 번째 경로 세그먼트**를 `modelDir` 로, 그 앞까지를 리소스 베이스로 쓴다. 즉 model3.json 은 **어떤 디렉토리 바로 안**에 있어야 한다. 베이스 바로 밑에 model3.json 만 덜렁 두면 안 된다.

**파일명은 ASCII 권장.** 한글·중국어 파일명(예: `运行文件.model3.json`)도 퍼센트 인코딩 왕복이 성립해 동작하지만, 디버깅이 번거로워진다.

---

## 3. model3.json 에서 읽어야 할 것

모델 폴더의 `*.model3.json` 에서 아래 네 군데만 보면 된다.

| 위치 | 왜 필요한가 |
|---|---|
| `FileReferences.Expressions[]` | `emotionMap` 값의 근거. 배열 **순서가 곧 인덱스**, 각 항목의 `Name` 이 문자열 지정 시 쓰는 이름 |
| `FileReferences.Motions` | 그룹 이름 확인. idle 모션과 탭 모션에 쓰인다 |
| `HitAreas[].Name` | `tapMotions` 의 키와 맞춰야 한다 |
| `Groups` 의 `EyeBlink` / `LipSync` | 눈 깜빡임과 입 동기화. 없으면 입이 안 움직인다 |

확인용 명령 (jq 필요):

```sh
MODEL=BE/live2d-models/mymodel/mymodel.model3.json
jq '.FileReferences.Expressions | to_entries | map({index: .key, name: .value.Name})' "$MODEL"
jq '.FileReferences.Motions | to_entries | map({group: .key, count: (.value|length)})' "$MODEL"
jq '.HitAreas' "$MODEL"
jq '.Groups | map(.Name)' "$MODEL"
```

### 주의: Expressions 가 아예 없는 모델이 있다

예를 들어 동봉된 `shizuku` 는 `FileReferences` 에 `Expressions` 키 자체가 없다. 이런 모델은 **표정 전환이 불가능**하다. `emotionMap` 에 숫자를 넣어도 아무 일도 일어나지 않는다. 표정이 필요하면 Expressions 가 있는 모델을 쓰거나, `.exp3.json` 파일을 만들어 model3.json 에 등록해야 한다.

### 주의: exp3 파일이 있어도 등록 안 됐으면 무의미하다

모델 폴더에 `*.exp3.json` 이 굴러다녀도 `FileReferences.Expressions` 배열에 들어있지 않으면 SDK 가 로드하지 않는다. 폴더 목록이 아니라 **반드시 model3.json 안의 배열**을 기준으로 삼을 것.

### VTube Studio 배포 모델을 붙일 때 (중요)

BOOTH 등에서 파는 VTS 용 모델은 표정을 `.vtube.json` 의 Hotkeys 로 관리하므로, **model3.json 에는 `Expressions` 키가 아예 없고 `LipSync` 의 `Ids` 가 빈 배열인 경우가 많다.** 그대로 쓰면 표정도 안 되고 입도 안 움직인다. 아래 순서로 직접 채워야 한다.

1. **표정 파일의 의도를 파악한다.** 파일명만으로는 알기 어려우므로 `.vtube.json` 의 `Hotkeys` 를 본다. `Name` 이 제작자가 붙인 설명이고 `File` 이 대응하는 exp3 다.
   ```sh
   jq '.Hotkeys | map({name: .Name, file: .File, action: .Action})' 모델.vtube.json
   ```
2. **감정용과 소품/포즈용을 가른다.** 모자 벗기, 가방 내리기, 팔 올리기 같은 토글은 감정이 아니므로 `Expressions` 에 등록은 하되 `emotionMap` 에는 넣지 않는다.
3. **중립 표정을 만든다.** VTS 모델은 "전체 해제" 핫키로 원상복귀하므로 중립 exp3 가 없다. `Parameters` 가 빈 배열인 `neutral.exp3.json` 을 만들어 등록하고 `defaultEmotion` 으로 지정한다.
4. **LipSync 를 채운다.** moc3 에서 입 파라미터 ID 를 찾아 `Groups` 의 `LipSync.Ids` 에 넣는다. 보통 `ParamMouthOpenY` 다.
   ```sh
   strings -n 3 모델.moc3 | grep -iE "mouth|eye|brow" | sort -u
   ```
5. `EyeBlink` 그룹도 비어 있으면 `ParamEyeLOpen` / `ParamEyeROpen` 을 넣는다.

표정 Blend 가 `Add` 여도 상관없다. SDK 는 표정을 교체 적용하므로 누적되지 않는다.

> 실제 적용 사례: `live2d-models/MIA` (VTS 배포 모델). Expressions 11개를 신규 등록하고 `neutral.exp3.json` 을 추가, `LipSync.Ids` 에 `ParamMouthOpenY` 를 채워 동작시켰다.

---

## 4. model_dict.json 엔트리 작성

최상위는 **배열(list)** 이다. 객체로 감싸면 안 된다.

```json
[
  {
    "name": "mymodel",
    "url": "/live2d-models/mymodel/mymodel.model3.json",
    "kScale": 0.5,
    "initialXshift": 0,
    "initialYshift": 0,
    "defaultEmotion": 0,
    "emotionMap": {
      "기쁨": 2,
      "슬픔": 4,
      "화남": "angry_face"
    },
    "tapMotions": {
      "Head": { "TapHead": 1 },
      "Body": { "TapBody": 1 }
    }
  }
]
```

### 4-1. 필드 표

| 필드 | 필수 | 타입 | 의미 |
|---|---|---|---|
| `name` | ✅ | string | 백엔드 매칭 키. `conf.yaml` 의 `live2d_model_name` 과 정확히 일치해야 한다 |
| `url` | ✅ | string | `/live2d-models/...` 로 시작하는 서버 상대 경로 |
| `kScale` | ✅ | number | 모델 크기. 프론트에서 내부적으로 2배 저장 후 되돌리는 보정이 있으므로 **0.5 전후에서 시작**해 눈으로 맞출 것 |
| `initialXshift` | ✅ | number | 가로 오프셋(논리 픽셀). 화면 DPR 이 곱해진다 |
| `initialYshift` | ✅ | number | 세로 오프셋(논리 픽셀) |
| `emotionMap` | ✅ | object | 감정 키워드 → Expressions 인덱스(number) 또는 Name(string). **없으면 기동 중 KeyError** |
| `defaultEmotion` | ⬜ | number \| string | AI 가 말을 마치고 idle 로 돌아갈 때 복귀할 표정 |
| `tapMotions` | ⬜ | object | `{ "<HitArea Name>": { "<모션 그룹명>": 가중치 } }` |
| `pointerInteractive` | ⬜ | boolean | `false` 면 탭 모션 비활성. 설정 사이드바에도 토글이 있다 |
| `scrollToResize` | ⬜ | boolean | 휠로 크기 조절 허용 여부 |
| `screenPosition` | ⬜ | object | `{x, y}` 정규화 좌표. 기본 `{x:0.85, y:0.85}` |
| `eyeTopRatio` | ⬜ | number | 캔버스 세로 기준 눈 위치(0=위, 1=아래). 기본 0.2 |
| `eyeLeftRatio` | ⬜ | number | 캔버스 가로 기준 눈 위치(0=좌, 1=우). 기본 0.5 |

### 4-2. 적어도 무시되는 필드

아래는 기존 파일에 들어있더라도 **읽는 코드가 없다.** 새로 쓸 때 넣지 말 것.

- `kXOffset` — 레포 전체에서 소비처 없음
- `idleMotionGroupName` — 타입 선언만 있고 읽히지 않음. idle 그룹명은 SDK 에 `'Idle'` 로 하드코딩돼 있다 (`FE/src/renderer/WebSDK/src/lappdefine.ts`)
- `description` — 메모용
- `initialScale` — 타입에만 존재

### 4-3. `defaultEmotion: -1` 함정

`-1` 은 `undefined` 가 아니므로 "기본 표정 사용" 분기를 타지 않는다. `getExpressionName(-1)` 이 빈 문자열을 반환해서 **표정 복귀가 그냥 안 일어난다.** 표정 리셋을 끄고 싶은 의도라면 그대로 둬도 되지만, 복귀를 원하면 실제 인덱스나 `Name` 을 넣어야 한다.

---

## 5. emotionMap 과 감정 키워드

### 5-1. 감정 키워드 목록의 유일한 출처는 emotionMap 의 키다

`conf.yaml` 이 아니다. 백엔드가 `emotionMap` 키들을 모아 `[키1], [키2], ...` 형태 문자열로 만들고, 시스템 프롬프트의 `[<insert_emomap_keys>]` 자리에 치환해 넣는다.

- 생성: `BE/src/open_llm_vtuber/live2d_model.py` (`emo_map`, `emo_str`)
- 주입: `BE/src/open_llm_vtuber/service_context.py` 의 `[<insert_emomap_keys>]` 치환
- 템플릿: `BE/prompts/utils/live2d_expression_prompt.txt`
- `conf.yaml` 은 `tool_prompts.live2d_expression_prompt` 로 **파일 이름만** 지정한다

즉 **모델을 바꾸면 LLM 이 쓸 수 있는 표정 키워드도 같이 바뀐다.** 캐릭터 프롬프트에 표정 목록을 하드코딩해 뒀다면 함께 고쳐야 한다.

### 5-2. 값 해석 규칙

- **숫자** → `FileReferences.Expressions` 배열의 0-based 인덱스. 인덱스로 `Name` 을 역조회한 뒤 적용한다.
- **문자열** → `Name` 을 직접 지정.

둘을 한 맵 안에서 섞어 써도 된다. 유지보수성만 보면 **문자열(Name) 지정이 안전하다.** 모델 쪽에서 Expressions 순서가 바뀌어도 깨지지 않기 때문이다.

### 5-3. 키 명명 규칙

- 키는 내부적으로 소문자 정규화되어 비교된다. 대소문자는 구분하지 않는다.
- 파싱은 정규식이 아니라 `[` 를 찾아 키 접두사를 선형 매칭하는 방식이다. 따라서 **한 키가 다른 키의 접두사가 되지 않게** 하는 편이 안전하다 (예: `기쁨` 과 `기쁨2` 를 같이 두지 말 것).
- 한국어 키를 써도 된다. 실제로 그렇게 쓰고 있다.

### 5-4. 한 문장에 여러 표정을 써도 첫 번째만 적용된다

프론트가 추출된 표정 배열에서 **첫 원소만** 사용한다 (`FE/src/renderer/src/hooks/utils/use-audio-task.ts`). 문장마다 표정 하나씩 쓰는 것을 전제로 설계돼 있다.

### 5-5. emotionMap 과 무관한 태그들

아래 태그는 백엔드에 하드코딩돼 있어 모델과 상관없이 항상 쓸 수 있다 (`BE/src/open_llm_vtuber/live2d_model.py` 상단 상수).

- 시선: `[look_left]`, `[look_up]` 등 9종 + `[focus:x,y]`
- 이동: `[move_left]` 등 9종 (상대 누적, `[move_center]` 만 절대 리셋)
- 크기: `[scale_up]`, `[scale_down]`, `[scale_normal]`
- 음량: `[volume_up]`, `[volume_down]`

---

## 6. 모션 설정

### 6-1. idle 모션

`model3.json` 의 `FileReferences.Motions` 에 **정확히 `"Idle"`** 이라는 그룹명이 있어야 자동 재생된다. `idleMotionGroupName` 필드로 바꿀 수 없다. 그룹명이 다르면 model3.json 쪽을 고칠 것.

`Motions` 가 빈 객체 `{}` 인 모델은 idle 모션도 탭 모션도 없다. 물리 연산과 눈 깜빡임만 동작한다.

### 6-2. 탭 모션

`tapMotions` 의 **키는 `HitAreas[].Name`** 과 일치해야 한다.

```
tapMotions: { "<HitArea Name>": { "<Motions 그룹명>": 가중치 } }
```

동작 순서:
1. 탭된 히트 영역 이름이 `tapMotions` 에 있으면 그 영역의 모션 중 가중치 랜덤 선택
2. 없으면 **전체 영역의 가중치를 합산**해 랜덤 선택 (fallback)
3. 아무것도 없으면 무반응

즉 **이름이 틀려도 조용히 fallback 으로 넘어가서** 오동작을 눈치채기 어렵다. 반드시 `jq '.HitAreas' model3.json` 으로 실제 이름을 확인하고 적을 것. `HitAreas` 의 `Name` 이 빈 문자열인 모델도 흔하다.

---

## 7. conf.yaml 연결

```yaml
character_config:
  live2d_model_name: 'mymodel'   # model_dict.json 의 name 과 완전히 동일
```

- 스키마상 **필수 항목**이다 (`BE/src/open_llm_vtuber/config_manager/character.py`).
- 이름이 맞지 않으면 기동 중 `KeyError` 와 함께 critical 로그가 뜬다.
- 캐릭터 프리셋별로 다른 모델을 쓰려면 `BE/characters/*.yaml` 안에서 `live2d_model_name` 을 지정하면 된다 (`BE/characters/README.md` 참고).

---

## 8. 눈 위치 보정 (선택)

`eyeTopRatio` / `eyeLeftRatio` 를 적지 않으면 백엔드가 **moc3 바이너리를 직접 파싱해 자동 추정**한다. 실패해도 경고만 남기고 기본값(0.2 / 0.5)으로 진행한다.

수동으로 재려면:

```sh
python BE/tools/detect_eye_ratio.py <모델 경로>
```

이 값은 화면 위 캐릭터 눈 높이를 기준으로 하는 기능(시선·카메라 정렬 등)에 쓰인다. 모델이 화면에서 유난히 위나 아래를 보는 느낌이면 여기를 의심할 것.

---

## 9. 반영과 캐시

- `model_dict.json` 은 **백엔드 기동 시 1회만** 읽힌다. 핫리로드 없음. 고쳤으면 재시작.
- 프론트는 `modelInfo` 를 **localStorage 에 캐시**한다 (`FE/src/renderer/src/context/live2d-config-context.tsx`). 백엔드를 고쳤는데 화면이 그대로면 localStorage 의 `modelInfo` 키를 지우고 새로고침.

---

## 10. 검증 체크리스트

새 모델을 붙인 뒤 아래를 순서대로 확인한다.

- [ ] 백엔드가 critical 로그 없이 기동하는가 (`live2d_model_name` 불일치, `emotionMap` 누락이 여기서 걸린다)
- [ ] 브라우저에 모델이 보이는가 (안 보이면 `url` 경로, 디렉토리 깊이, localStorage 순서로 확인)
- [ ] 크기와 위치가 맞는가 (`kScale`, `initialXshift`, `initialYshift`)
- [ ] 입이 소리에 맞춰 움직이는가 (`Groups` 의 `LipSync`)
- [ ] 눈을 깜빡이는가 (`Groups` 의 `EyeBlink`)
- [ ] `emotionMap` 의 키를 하나씩 대사에 넣어 보고 표정이 실제로 바뀌는가
- [ ] 말이 끝난 뒤 표정이 `defaultEmotion` 으로 돌아가는가 (`-1` 이면 안 돌아가는 게 정상)
- [ ] 캐릭터를 클릭했을 때 모션이 나오는가 (`tapMotions`, `HitAreas[].Name`)
- [ ] 가만히 둘 때 idle 모션이 도는가 (`Motions` 의 `"Idle"` 그룹)

---

## 11. 증상별 원인 표

| 증상 | 1순위 원인 |
|---|---|
| 기동 시 `KeyError` | `live2d_model_name` 과 `name` 불일치, 또는 `emotionMap` 누락 |
| 모델이 안 보임 | `url` 경로 오류 / model3.json 이 디렉토리 바로 안에 없음 / localStorage 캐시 |
| 표정이 안 바뀜 | `Expressions` 배열이 없거나, 인덱스 범위를 벗어남 |
| 특정 표정만 안 바뀜 | 그 인덱스의 exp3 파일 경로가 깨졌거나 Name 오타 |
| 말 끝나고 표정이 안 풀림 | `defaultEmotion` 이 `-1` 또는 미지정 |
| 클릭해도 반응 없음 | `pointerInteractive: false`, 또는 `Motions` 가 비어 있음 |
| 클릭하면 엉뚱한 모션 | `tapMotions` 키가 `HitAreas[].Name` 과 불일치해 fallback 발동 |
| idle 모션 없음 | `Motions` 그룹명이 `"Idle"` 이 아님 |
| 입이 안 움직임 | `Groups` 에 `LipSync` 없음 |
| 모델 바꿨는데 그대로 | 백엔드 미재시작 또는 localStorage `modelInfo` 잔존 |

---

## 12. 라이선스 주의

`live2d-models/` 는 `.gitignore` 에서 **디렉토리 전체가 제외**되고, 공식 샘플인 `mao_pro` 와 `shizuku` 만 예외로 추적된다.

유료로 구매했거나 재배포가 금지된 모델을 추가할 때는 이 예외 목록을 **건드리지 말 것.** 새 모델 폴더를 추적 대상에 넣으면 그대로 공개 저장소에 올라간다.
