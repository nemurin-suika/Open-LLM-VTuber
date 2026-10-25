#!/bin/bash
# =============================================================================
# bootstrap.sh — 새 맥에 이 스택을 통째로 깔아 주는 설치 스크립트
# =============================================================================
#
# 전제: 이 저장소(백엔드)를 먼저 클론한 상태에서 레포 루트 기준으로 실행한다.
#
#   git clone https://github.com/nemurin-suika/Open-LLM-VTuber.git \
#       -b v1-release-nemurin ~/develop/Open-LLM-VTuber-v1.2.1-en
#   cd ~/develop/Open-LLM-VTuber-v1.2.1-en
#   ./scripts/bootstrap.sh
#
# 하는 일:
#   1. 필요한 도구 점검 (없으면 brew 로 설치할지 물어본다)
#   2. letta / 프론트엔드 저장소 클론
#   3. frontend 심볼릭 링크 연결
#   4. 이 맥에 맞게 절대경로·주소 치환
#   5. PostgreSQL 데이터 디렉토리 초기화
#   6. 의존성 설치 (uv sync / npm install)
#   7. conf.yaml 생성
#   8. 손으로 옮겨야 하는 것들 안내
#
# 안 하는 일 (일부러):
#   - launchd 등록. 상주 운영 설정은 환경마다 다르므로 손대지 않는다.
#   - Live2D 모델·TTS 참조 음성 설치. 라이선스 때문에 저장소에 없다.
#   - API 키 입력. ~/.secrets 템플릿만 만들어 둔다.
#
# 옵션:
#   --dir <경로>   저장소를 모아 둘 상위 디렉토리 (기본: 이 저장소의 부모)
#   --yes          되묻지 않고 전부 진행 (brew 설치 포함)
#   --skip-brew    패키지 설치를 건너뛴다 (점검만)
#   --skip-npm     프론트엔드 의존성 설치를 건너뛴다
# =============================================================================

set -u
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="$(dirname "$REPO_ROOT")"
ASSUME_YES=0
SKIP_BREW=0
SKIP_NPM=0

LETTA_REPO="https://github.com/nemurin-suika/letta.git"
LETTA_BRANCH="nemurin"
LETTA_DIRNAME="letta"

WEB_REPO="https://github.com/nemurin-suika/Open-LLM-VTuber-Web.git"
WEB_BRANCH="nemurin"
WEB_DIRNAME="Open-LLM-VTuber-Web"

BREW_PACKAGES=(postgresql@17 redis node ollama)

while [ $# -gt 0 ]; do
    case "$1" in
        --dir)       BASE_DIR="$2"; shift 2 ;;
        --yes|-y)    ASSUME_YES=1; shift ;;
        --skip-brew) SKIP_BREW=1; shift ;;
        --skip-npm)  SKIP_NPM=1; shift ;;
        -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "모르는 옵션: $1" >&2; exit 1 ;;
    esac
done

# ── 출력 도우미 ───────────────────────────────────────────────────────────────
STEP=0
step()  { STEP=$((STEP + 1)); echo; echo "── [$STEP] $*"; }
info()  { echo "   $*"; }
warn()  { echo "   ! $*" >&2; }
die()   { echo; echo "ERROR: $*" >&2; exit 1; }

confirm() {
    [ "$ASSUME_YES" = "1" ] && return 0
    if [ ! -t 0 ]; then
        warn "비대화 실행이라 '아니오'로 처리합니다: $1"
        return 1
    fi
    printf "   %s [y/N] " "$1"
    read -r _ans
    case "$_ans" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

clone_or_update() {
    local url="$1" branch="$2" dest="$3" label="$4"
    if [ -d "$dest/.git" ]; then
        info "$label: 이미 있음 ($dest) — 클론 건너뜀"
        local cur
        cur="$(git -C "$dest" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        [ "$cur" != "$branch" ] && warn "$label 의 현재 브랜치가 '$cur' 입니다 (기대: $branch)"
        return 0
    fi
    if [ -e "$dest" ]; then
        die "$dest 가 git 저장소가 아닌 채로 이미 존재합니다. 치우고 다시 실행하세요."
    fi
    info "$label: 클론 중 → $dest"
    git clone --branch "$branch" "$url" "$dest" || die "$label 클론 실패"
}

echo "============================================================"
echo " Open-LLM-VTuber 스택 설치"
echo "============================================================"
info "백엔드 저장소 : $REPO_ROOT"
info "상위 디렉토리 : $BASE_DIR"

[ "$(uname -s)" = "Darwin" ] || warn "macOS 기준으로 만든 스크립트입니다. 다른 OS 에서는 확인이 필요합니다."

# ── 1. 도구 점검 ──────────────────────────────────────────────────────────────
step "필요한 도구 점검"

command -v git >/dev/null 2>&1 || die "git 이 없습니다. Xcode Command Line Tools 를 먼저 설치하세요."
command -v python3 >/dev/null 2>&1 || die "python3 가 없습니다."

if ! command -v brew >/dev/null 2>&1; then
    warn "Homebrew 가 없습니다. 패키지 설치는 건너뜁니다."
    warn "  설치: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    SKIP_BREW=1
fi

if [ "$SKIP_BREW" = "0" ]; then
    missing=()
    for pkg in "${BREW_PACKAGES[@]}"; do
        if brew list --versions "$pkg" >/dev/null 2>&1; then
            info "OK   $pkg"
        else
            info "없음 $pkg"
            missing+=("$pkg")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        if confirm "brew install ${missing[*]} 를 실행할까요?"; then
            brew install "${missing[@]}" || die "brew install 실패"
        else
            warn "설치를 건너뜁니다. 나중에 직접 설치해야 기동됩니다."
        fi
    fi
else
    for pkg in "${BREW_PACKAGES[@]}"; do
        info "점검만: $pkg"
    done
fi

if ! command -v uv >/dev/null 2>&1; then
    if confirm "uv (파이썬 패키지 매니저) 가 없습니다. 설치할까요?"; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv 설치 실패"
        export PATH="$HOME/.local/bin:$PATH"
    else
        die "uv 없이는 진행할 수 없습니다."
    fi
fi
info "OK   uv ($(uv --version 2>/dev/null))"

if command -v claude >/dev/null 2>&1; then
    info "OK   claude CLI"
else
    warn "claude CLI 가 없습니다. letta 의 claude_cli 모델을 쓰려면 Claude Code CLI 설치와 로그인이 필요합니다."
fi

# ── 2. 저장소 클론 ────────────────────────────────────────────────────────────
step "저장소 준비"
mkdir -p "$BASE_DIR"
LETTA_DIR="$BASE_DIR/$LETTA_DIRNAME"
WEB_DIR="$BASE_DIR/$WEB_DIRNAME"
clone_or_update "$LETTA_REPO" "$LETTA_BRANCH" "$LETTA_DIR" "letta"
clone_or_update "$WEB_REPO"   "$WEB_BRANCH"   "$WEB_DIR"   "프론트엔드"

# ── 3. frontend 심볼릭 링크 ───────────────────────────────────────────────────
step "frontend 링크 연결"
# 백엔드 run.sh 가 'pushd frontend && npm run build:web' 을 하므로
# 레포 루트의 frontend 가 프론트엔드 저장소를 가리켜야 한다. (.gitignore 대상)
LINK="$REPO_ROOT/frontend"
if [ -L "$LINK" ]; then
    cur="$(readlink "$LINK")"
    if [ "$cur" = "$WEB_DIR" ]; then
        info "이미 연결됨: frontend -> $WEB_DIR"
    else
        info "링크 대상 변경: $cur -> $WEB_DIR"
        rm -f "$LINK" && ln -s "$WEB_DIR" "$LINK"
    fi
elif [ -e "$LINK" ]; then
    warn "$LINK 가 링크가 아닌 실제 파일/디렉토리입니다. 손대지 않습니다."
else
    ln -s "$WEB_DIR" "$LINK"
    info "연결 완료: frontend -> $WEB_DIR"
fi

# ── 4. 이 맥에 맞게 경로·주소 치환 ────────────────────────────────────────────
step "로컬 환경에 맞게 경로 치환"
# letta/run.sh 에는 개발 환경의 절대경로와 LAN 주소가 들어 있다.
# 사용자명이 다르거나 로컬 전용으로 쓸 때는 바꿔 줘야 한다.
LETTA_RUN="$LETTA_DIR/run.sh"
if [ -f "$LETTA_RUN" ]; then
    changed=0
    # 개발 환경의 홈 디렉토리가 그대로 박혀 있으면 이 맥 기준으로 맞춘다.
    if grep -qE '/Users/[A-Za-z0-9._-]+/' "$LETTA_RUN"; then
        sed -i '' -E "s|/Users/[A-Za-z0-9._-]+/|${HOME}/|g" "$LETTA_RUN" && changed=1
        info "letta/run.sh: 홈 디렉토리 경로를 $HOME 으로 치환"
    fi
    if grep -q 'VTUBER_APPROVAL_URL=http://192\.168\.' "$LETTA_RUN"; then
        sed -i '' "s|VTUBER_APPROVAL_URL=http://192\.168\.[0-9.]*:|VTUBER_APPROVAL_URL=http://127.0.0.1:|" "$LETTA_RUN" && changed=1
        info "letta/run.sh: 승인 서버 주소를 127.0.0.1 로 치환"
    fi
    # 백엔드 경로가 이 저장소와 다른 이름일 수 있으므로 한 번 더 맞춘다.
    if grep -q 'VTUBER_CLAUDE_HOOK_SETTINGS=' "$LETTA_RUN"; then
        sed -i '' "s|VTUBER_CLAUDE_HOOK_SETTINGS=.*|VTUBER_CLAUDE_HOOK_SETTINGS=$REPO_ROOT/scripts/claude_hook_settings.json|" "$LETTA_RUN" && changed=1
        info "letta/run.sh: 훅 설정 경로를 이 저장소로 맞춤"
    fi
    [ "$changed" = "0" ] && info "치환할 것이 없습니다."
    # 치환 결과가 커밋으로 섞이지 않게 알려 준다.
    if ! git -C "$LETTA_DIR" diff --quiet 2>/dev/null; then
        warn "letta 저장소에 로컬 수정이 생겼습니다. 이 변경은 커밋하지 마세요."
    fi
else
    warn "$LETTA_RUN 이 없습니다. letta 클론을 확인하세요."
fi

# ── 5. 비밀값 파일 ────────────────────────────────────────────────────────────
step "비밀값 파일 (~/.secrets)"
# letta/run.sh 가 첫 줄에서 source ~/.secrets 를 한다. 없으면 기동이 멈춘다.
if [ -f "$HOME/.secrets" ]; then
    info "이미 있음 — 손대지 않습니다."
else
    cat > "$HOME/.secrets" <<'SECRETS'
# letta/run.sh 가 읽는 비밀값 파일. 이 파일은 저장소에 넣지 않는다.
# 쓰지 않는 항목은 빈 값으로 두어도 된다.

# 임베딩용 Gemini API 키 (여러 개 넣으면 429 때 차례로 넘어간다)
GEMINI_KEYS=()
SECRETS
    chmod 600 "$HOME/.secrets"
    info "템플릿 생성: ~/.secrets (필요한 키를 채우세요)"
fi

# ── 6. PostgreSQL 데이터 디렉토리 ─────────────────────────────────────────────
step "PostgreSQL 데이터 디렉토리"
PG_DATA="${MIZUKI_PG_DATA:-$HOME/.mizuki/pgdata}"
PG_BIN="/opt/homebrew/opt/postgresql@17/bin"
if [ -f "$PG_DATA/PG_VERSION" ]; then
    info "이미 초기화됨: $PG_DATA"
elif [ -x "$PG_BIN/initdb" ]; then
    mkdir -p "$(dirname "$PG_DATA")"
    info "initdb 실행: $PG_DATA"
    # 로케일을 명시하지 않으면 launchd 기동 시 postgres 17 이 즉시 죽는다.
    "$PG_BIN/initdb" -D "$PG_DATA" --encoding=UTF8 --locale=en_US.UTF-8 \
        >/dev/null || warn "initdb 실패 — 직접 확인이 필요합니다."
    [ -f "$PG_DATA/PG_VERSION" ] && info "초기화 완료"
else
    warn "$PG_BIN/initdb 가 없습니다. postgresql@17 설치 후 다시 실행하세요."
fi

# ── 7. 의존성 설치 ────────────────────────────────────────────────────────────
step "백엔드 의존성 설치 (uv sync)"
( cd "$REPO_ROOT" && uv sync ) || warn "백엔드 uv sync 실패 — 직접 확인하세요."

step "letta 의존성 설치 (uv sync)"
if [ -d "$LETTA_DIR" ]; then
    ( cd "$LETTA_DIR" && uv sync ) || warn "letta uv sync 실패 — 직접 확인하세요."
fi

step "프론트엔드 의존성 설치"
if [ "$SKIP_NPM" = "1" ]; then
    info "--skip-npm 지정됨 — 건너뜀"
elif command -v npm >/dev/null 2>&1; then
    ( cd "$WEB_DIR" && npm install ) || warn "npm install 실패 — 직접 확인하세요."
else
    warn "npm 이 없습니다. node 설치 후 '$WEB_DIR' 에서 npm install 을 돌리세요."
fi

# ── 8. 임베딩 모델 ────────────────────────────────────────────────────────────
step "임베딩 모델 (ollama bge-m3)"
if command -v ollama >/dev/null 2>&1; then
    if ollama list 2>/dev/null | grep -q 'bge-m3'; then
        info "이미 받아져 있음"
    elif confirm "ollama pull bge-m3 를 실행할까요? (수 분 걸립니다)"; then
        ollama pull bge-m3 || warn "ollama pull 실패 — 나중에 직접 받으세요."
    else
        info "건너뜀. 나중에 'ollama pull bge-m3' 를 실행하세요."
    fi
else
    warn "ollama 가 없습니다. 임베딩을 다른 provider 로 쓸 거라면 무시해도 됩니다."
fi

# ── 9. 런타임 디렉토리 ────────────────────────────────────────────────────────
step "런타임 디렉토리 준비"
# 서버가 기동할 때 정적 경로로 마운트하는 디렉토리들이다. 비어 있어도 되지만
# 존재는 해야 한다. 저장소에 비는 디렉토리는 남지 않으므로 여기서 만든다.
for d in cache live2d-models backgrounds avatars web_tool logs; do
    if [ -d "$REPO_ROOT/$d" ]; then
        info "OK   $d"
    else
        mkdir -p "$REPO_ROOT/$d" && info "생성 $d"
    fi
done

# ── 10. conf.yaml ─────────────────────────────────────────────────────────────
step "conf.yaml 생성"
if [ -f "$REPO_ROOT/conf.yaml" ]; then
    info "이미 있음 — 손대지 않습니다."
    info "다시 만들려면: python3 scripts/make_conf.py --force"
else
    if [ "$ASSUME_YES" = "1" ] || [ ! -t 0 ]; then
        python3 "$REPO_ROOT/scripts/make_conf.py" --defaults \
            || warn "conf.yaml 생성 실패"
    else
        python3 "$REPO_ROOT/scripts/make_conf.py" || warn "conf.yaml 생성 실패"
    fi
fi

# ── 11. 마무리 안내 ───────────────────────────────────────────────────────────
echo
echo "============================================================"
echo " 설치 절차 끝. 남은 건 손으로 옮겨야 하는 것들입니다."
echo "============================================================"
cat <<GUIDE

1) Live2D 모델
   라이선스 때문에 저장소에 들어 있지 않습니다. 모델 폴더를 통째로
     $REPO_ROOT/live2d-models/<모델이름>/
   에 넣고, 같은 폴더 기준으로 model_dict.json 에 항목을 추가하세요.
   붙이는 절차와 흔한 함정은 doc/LIVE2D_MODEL_SETUP.md 에 정리돼 있습니다.

2) 목소리
   기본값은 edge_tts 라 바로 소리가 납니다. 따로 학습한 목소리를 쓰려면
   TTS 서버를 띄운 뒤 conf.yaml 의 tts_model 을 gpt_sovits_tts 로 바꾸고,
   참조 음성 파일을 TTS 서버가 찾을 수 있는 위치에 두세요.

3) 로그인·키
   - claude CLI: 'claude auth login' 한 번 실행
   - ~/.secrets: 필요한 API 키 채우기

4) 페르소나
   conf.yaml 의 persona_prompt 가 자리표시 문구라면 직접 채우세요.
   파일로 준비해 두고 아래처럼 다시 만들어도 됩니다.
     python3 scripts/make_conf.py --persona <파일> --force

띄우기:
   ./scripts/start-all.sh          # letta 기동 → 준비되면 백엔드 기동
   브라우저에서 http://127.0.0.1:12393 접속

GUIDE
