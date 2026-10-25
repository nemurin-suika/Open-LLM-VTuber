#!/bin/bash
# =============================================================================
# start-all.sh — letta, 목소리, 백엔드를 순서대로 띄운다
# =============================================================================
#
#   ./scripts/start-all.sh              # letta·TTS 백그라운드 → 백엔드 포그라운드
#   ./scripts/start-all.sh --build      # 프론트엔드를 다시 빌드하고 띄운다
#   ./scripts/start-all.sh --stop       # 전부 정지
#   ./scripts/start-all.sh --status     # 상태만 본다
#   ./scripts/start-all.sh --no-tts     # 목소리 서버는 띄우지 않는다
#
# 왜 순서가 중요한가:
#   백엔드 run.sh 안의 letta 에이전트 설정 블록은 8283 포트에 한 번만 접속을
#   시도하고, 실패하면 조용히 건너뛴 채로 서버를 띄운다. 그러면 시스템
#   프롬프트와 메모리 블록 갱신이 누락된 상태로 돌아간다. 그래서 여기서
#   letta 의 health 가 ok 가 될 때까지 기다린 다음에 백엔드를 띄운다.
#
#   목소리(GPT-SoVITS)는 백엔드보다 늦게 떠도 대화는 되지만, 첫 몇 마디가
#   소리 없이 지나간다. 모델 올리는 데 시간이 걸리므로 미리 띄워 둔다.
#   conf.yaml 의 tts_model 이 gpt_sovits_tts 일 때만 띄운다.
#
# 상주 운영(로그인하면 자동 기동)을 원하면 launchd 로 따로 등록하면 된다.
# 이 스크립트는 터미널에서 직접 돌리는 용도다.
# =============================================================================

set -u
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="$(dirname "$REPO_ROOT")"
MIZUKI_AI_HOME="${MIZUKI_AI_HOME:-$HOME/ai}"
LETTA_DIR="${LETTA_REPO_DIR:-$BASE_DIR/letta}"
LOG_DIR="$MIZUKI_AI_HOME/.mizuki/logs"
LETTA_LOG="$LOG_DIR/letta.log"
LETTA_PID="$LOG_DIR/letta.pid"

LETTA_PORT="${LETTA_PORT:-8283}"
LETTA_HEALTH="http://127.0.0.1:${LETTA_PORT}/v1/health/"
LETTA_WAIT="${LETTA_WAIT:-180}"

BACKEND_PORT="${BACKEND_PORT:-12393}"

TTS_DIR="${TTS_REPO_DIR:-$BASE_DIR/GPT-SoVITS-v3lora-20250228}"
TTS_PORT="${TTS_PORT:-9880}"
TTS_WAIT="${TTS_WAIT:-300}"
TTS_LOG="$LOG_DIR/gpt-sovits.log"
TTS_PID="$LOG_DIR/gpt-sovits.pid"

BUILD=0
USE_TTS=1
ACTION="start"
for arg in "$@"; do
    case "$arg" in
        --build)  BUILD=1 ;;
        --stop)   ACTION="stop" ;;
        --status) ACTION="status" ;;
        --no-tts) USE_TTS=0 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "모르는 옵션: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

port_pid() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | tr '\n' ' '; }

# conf.yaml 에서 작은따옴표로 감싼 값 하나를 꺼낸다. 설정 한 줄을 보려고
# PyYAML 을 끌어오고 싶지 않아서 이렇게 한다.
#   conf_value <키> [블록이름]
# 블록 이름을 주면 그 줄부터 아래로만 찾는다. prompt_text 처럼 여러 TTS
# 블록에 같은 이름이 있는 키는 블록을 지정해야 엉뚱한 값을 집지 않는다.
conf_value() {
    local key="$1" from="${2:-}"
    local file="$REPO_ROOT/conf.yaml"
    [ -f "$file" ] || return 0
    {
        if [ -n "$from" ]; then
            sed -n "/^[[:space:]]*$from:/,\$p" "$file"
        else
            cat "$file"
        fi
    } | sed -n "s/^[[:space:]]*$key:[[:space:]]*'\(.*\)'.*/\1/p" \
      | grep -v '^[[:space:]]*$' | head -1
}

tts_enabled() {
    [ "$USE_TTS" = "1" ] || return 1
    [ "$(conf_value tts_model)" = "gpt_sovits_tts" ] || return 1
    return 0
}

status() {
    local lp bp tp
    lp="$(port_pid "$LETTA_PORT")"
    bp="$(port_pid "$BACKEND_PORT")"
    tp="$(port_pid "$TTS_PORT")"
    echo "letta  (:$LETTA_PORT)   : ${lp:-정지}"
    echo "목소리 (:$TTS_PORT)   : ${tp:-정지}"
    echo "백엔드 (:$BACKEND_PORT) : ${bp:-정지}"
    if [ -n "$lp" ]; then
        if curl -fs -m 3 "$LETTA_HEALTH" 2>/dev/null | grep -q '"status":"ok"'; then
            echo "letta health        : ok"
        else
            echo "letta health        : 응답 없음 (기동 중일 수 있음)"
        fi
    fi
}

stop_all() {
    local bp lp tp
    bp="$(port_pid "$BACKEND_PORT")"
    if [ -n "$bp" ]; then
        echo "[백엔드] 정지 ($bp)"
        # shellcheck disable=SC2086
        kill $bp 2>/dev/null
    fi
    tp="$(port_pid "$TTS_PORT")"
    if [ -n "$tp" ]; then
        echo "[목소리] 정지 ($tp)"
        # shellcheck disable=SC2086
        kill $tp 2>/dev/null
    fi
    rm -f "$TTS_PID"
    lp="$(port_pid "$LETTA_PORT")"
    if [ -n "$lp" ]; then
        echo "[letta] 정지 ($lp)"
        # shellcheck disable=SC2086
        kill $lp 2>/dev/null
    fi
    rm -f "$LETTA_PID"
    # 포트가 실제로 풀릴 때까지 잠깐 기다린다.
    for _ in $(seq 1 20); do
        [ -z "$(port_pid "$BACKEND_PORT")" ] && [ -z "$(port_pid "$LETTA_PORT")" ] && break
        sleep 0.5
    done
    echo "정지 완료."
}

case "$ACTION" in
    status) status; exit 0 ;;
    stop)   stop_all; exit 0 ;;
esac

# ── letta ────────────────────────────────────────────────────────────────────
if [ -n "$(port_pid "$LETTA_PORT")" ]; then
    echo "[letta] 이미 :$LETTA_PORT 에서 돌고 있습니다."
else
    [ -x "$LETTA_DIR/run.sh" ] || {
        echo "ERROR: $LETTA_DIR/run.sh 가 없습니다. LETTA_REPO_DIR 로 경로를 지정하거나 bootstrap.sh 를 먼저 돌리세요." >&2
        exit 1
    }
    echo "[letta] 기동 중... (로그: $LETTA_LOG)"
    ( cd "$LETTA_DIR" && exec ./run.sh --local ) >>"$LETTA_LOG" 2>&1 &
    echo $! > "$LETTA_PID"
fi

echo "[letta] 준비될 때까지 대기 (최대 ${LETTA_WAIT}s)"
_ok=0
_i=0
while [ "$_i" -lt "$LETTA_WAIT" ]; do
    if curl -fs -m 3 "$LETTA_HEALTH" 2>/dev/null | grep -q '"status":"ok"'; then
        _ok=1
        echo "[letta] 준비 완료 (${_i}s)"
        break
    fi
    _i=$((_i + 1))
    sleep 1
done

if [ "$_ok" -ne 1 ]; then
    echo "ERROR: letta 가 ${LETTA_WAIT}s 안에 올라오지 않았습니다." >&2
    echo "       로그를 확인하세요: tail -n 100 $LETTA_LOG" >&2
    exit 1
fi

# ── 목소리 (GPT-SoVITS) ──────────────────────────────────────────────────────
if ! tts_enabled; then
    if [ "$USE_TTS" = "0" ]; then
        echo "[목소리] --no-tts 지정됨 — 건너뜀"
    else
        echo "[목소리] conf.yaml 의 tts_model 이 gpt_sovits_tts 가 아닙니다 — 건너뜀"
    fi
elif [ -n "$(port_pid "$TTS_PORT")" ]; then
    echo "[목소리] 이미 :$TTS_PORT 에서 돌고 있습니다."
else
    if [ ! -x "$TTS_DIR/run-mac.sh" ]; then
        echo "[목소리] $TTS_DIR/run-mac.sh 가 없습니다 — 건너뜁니다." >&2
        echo "         TTS_REPO_DIR 로 경로를 지정하거나 bootstrap.sh 를 먼저 돌리세요." >&2
    else
        # 참조 음성과 그 대사는 conf.yaml 을 기준으로 맞춘다. 양쪽이 어긋나면
        # 백엔드가 보낸 참조음성 이름을 TTS 서버가 못 찾아서 소리가 안 난다.
        _ref="$(conf_value ref_audio_path gpt_sovits_tts)"
        _pt="$(conf_value prompt_text gpt_sovits_tts)"
        echo "[목소리] 기동 중... (로그: $TTS_LOG)"
        [ -n "$_ref" ] && echo "         참조음성: $_ref"
        (
            cd "$TTS_DIR" || exit 1
            [ -n "$_ref" ] && export GPTSOVITS_REF_AUDIO="$_ref"
            [ -n "$_pt" ] && export GPTSOVITS_PROMPT_TEXT="$_pt"
            export GPTSOVITS_PORT="$TTS_PORT"
            exec ./run-mac.sh
        ) >>"$TTS_LOG" 2>&1 &
        echo $! > "$TTS_PID"

        echo "[목소리] 준비될 때까지 대기 (최대 ${TTS_WAIT}s)"
        _tok=0
        _i=0
        while [ "$_i" -lt "$TTS_WAIT" ]; do
            if [ -n "$(port_pid "$TTS_PORT")" ]; then
                _tok=1
                echo "[목소리] 준비 완료 (${_i}s)"
                break
            fi
            _i=$((_i + 1))
            sleep 1
        done
        # 여기서 멈추지는 않는다. 목소리가 없어도 대화는 되고, 뒤늦게 떠도
        # 백엔드가 매 요청마다 다시 붙으므로 그때부터 소리가 난다.
        [ "$_tok" -ne 1 ] && echo "[목소리] ${TTS_WAIT}s 안에 올라오지 않았습니다. 로그: tail -n 100 $TTS_LOG" >&2
    fi
fi

# ── 백엔드 ───────────────────────────────────────────────────────────────────
if [ -n "$(port_pid "$BACKEND_PORT")" ]; then
    echo "ERROR: :$BACKEND_PORT 가 이미 사용 중입니다 (pid $(port_pid "$BACKEND_PORT"))." >&2
    echo "       먼저 './scripts/start-all.sh --stop' 으로 정리하세요." >&2
    exit 1
fi

cd "$REPO_ROOT" || exit 1

# 빌드 산출물이 없으면 백엔드가 frontend 디렉토리를 그대로 서빙해서 빈 화면이
# 뜬다. 처음 띄우는 것이라면 빌드를 건너뛰면 안 된다.
if [ ! -d "$REPO_ROOT/frontend/dist/web" ]; then
    echo "[프론트] 빌드 산출물이 없습니다 → 이번에는 빌드하고 띄웁니다."
    BUILD=1
fi

echo "[백엔드] 기동 — http://127.0.0.1:${BACKEND_PORT}"
if [ "$BUILD" = "1" ]; then
    exec ./run.sh
else
    # 프론트엔드를 고치지 않았다면 빌드를 건너뛰는 쪽이 훨씬 빠르다.
    exec ./run.sh --skip-build
fi
