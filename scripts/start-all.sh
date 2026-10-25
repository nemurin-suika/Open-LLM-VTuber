#!/bin/bash
# =============================================================================
# start-all.sh — letta 와 백엔드를 순서대로 띄운다
# =============================================================================
#
#   ./scripts/start-all.sh              # letta 백그라운드 → 백엔드 포그라운드
#   ./scripts/start-all.sh --build      # 프론트엔드를 다시 빌드하고 띄운다
#   ./scripts/start-all.sh --stop       # 양쪽 모두 정지
#   ./scripts/start-all.sh --status     # 상태만 본다
#
# 왜 순서가 중요한가:
#   백엔드 run.sh 안의 letta 에이전트 설정 블록은 8283 포트에 한 번만 접속을
#   시도하고, 실패하면 조용히 건너뛴 채로 서버를 띄운다. 그러면 시스템
#   프롬프트와 메모리 블록 갱신이 누락된 상태로 돌아간다. 그래서 여기서
#   letta 의 health 가 ok 가 될 때까지 기다린 다음에 백엔드를 띄운다.
#
# 상주 운영(로그인하면 자동 기동)을 원하면 launchd 로 따로 등록하면 된다.
# 이 스크립트는 터미널에서 직접 돌리는 용도다.
# =============================================================================

set -u
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="$(dirname "$REPO_ROOT")"
LETTA_DIR="${LETTA_REPO_DIR:-$BASE_DIR/letta}"
LOG_DIR="$HOME/.mizuki/logs"
LETTA_LOG="$LOG_DIR/letta.log"
LETTA_PID="$LOG_DIR/letta.pid"

LETTA_PORT="${LETTA_PORT:-8283}"
LETTA_HEALTH="http://127.0.0.1:${LETTA_PORT}/v1/health/"
LETTA_WAIT="${LETTA_WAIT:-180}"

BACKEND_PORT="${BACKEND_PORT:-12393}"

BUILD=0
ACTION="start"
for arg in "$@"; do
    case "$arg" in
        --build)  BUILD=1 ;;
        --stop)   ACTION="stop" ;;
        --status) ACTION="status" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "모르는 옵션: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

port_pid() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | tr '\n' ' '; }

status() {
    local lp bp
    lp="$(port_pid "$LETTA_PORT")"
    bp="$(port_pid "$BACKEND_PORT")"
    echo "letta  (:$LETTA_PORT)   : ${lp:-정지}"
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
    local bp lp
    bp="$(port_pid "$BACKEND_PORT")"
    if [ -n "$bp" ]; then
        echo "[백엔드] 정지 ($bp)"
        # shellcheck disable=SC2086
        kill $bp 2>/dev/null
    fi
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
