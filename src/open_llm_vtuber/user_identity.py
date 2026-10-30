"""접속자 신원 (TODO2 1-10 멀티 사용자 대응).

앞단 nginx + oauth2-proxy + authz 가 인증을 끝낸 뒤 업스트림(이 백엔드)으로
아래 헤더를 붙여 보낸다. 위조 방지는 nginx `auth-gate.conf` 에서 클라이언트가
보낸 동일 이름 헤더를 전부 빈 값으로 덮어쓰는 방식으로 이미 처리돼 있다.

    X-Mizuki-Email    로그인한 구글 계정 메일
    X-Mizuki-Name     표시 이름
    X-Mizuki-Role     서비스별 등급 (mizuki = admin / guest)
    X-Mizuki-Service  서비스 slug

헤더가 하나도 없으면 nginx 를 거치지 않은 접속이다. 그 경로는 LAN 안에서만
열려 있으므로(라우터가 443 만 포워딩) 네무린 본인으로 간주한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from loguru import logger

# 주인 계정. authz 의 ADMIN_EMAIL 과 같은 값이어야 한다.
OWNER_EMAIL = (os.environ.get("MIZUKI_OWNER_EMAIL") or "cxz124@gmail.com").lower()
OWNER_NAME = os.environ.get("MIZUKI_OWNER_NAME") or "네무린"

_HEADER_EMAIL = "x-mizuki-email"
_HEADER_NAME = "x-mizuki-name"
_HEADER_ROLE = "x-mizuki-role"
_HEADER_SERVICE = "x-mizuki-service"


@dataclass
class Identity:
    """접속자 한 명. `is_owner` 가 아니면 전부 손님으로 취급한다."""

    email: str = OWNER_EMAIL
    name: str = OWNER_NAME
    role: str = "admin"
    service: str = "mizuki"
    # 신원 헤더 없이 들어온 로컬(LAN) 접속인지
    local: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_owner(self) -> bool:
        return bool(self.email) and self.email.lower() == OWNER_EMAIL

    @property
    def is_guest(self) -> bool:
        return not self.is_owner

    @property
    def key(self) -> str:
        """세션 소유자 키. 주인은 빈 문자열(기존 세션 하위호환)."""
        return "" if self.is_owner else self.email.lower()

    @property
    def label(self) -> str:
        """프롬프트·로그에 쓰는 표시 문자열."""
        if self.is_owner:
            return OWNER_NAME
        return self.name or self.email.split("@")[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "service": self.service,
            "local": self.local,
            "is_owner": self.is_owner,
            "label": self.label,
        }


OWNER_IDENTITY = Identity()


def extract_identity(headers: Optional[Mapping[str, str]]) -> Identity:
    """요청 헤더에서 접속자 신원을 뽑는다. 실패해도 예외를 던지지 않는다."""
    if not headers:
        return Identity()

    def _get(name: str) -> str:
        try:
            return str(headers.get(name) or "").strip()
        except Exception:
            return ""

    email = _get(_HEADER_EMAIL).lower()
    if not email:
        # nginx 를 거치지 않은 로컬 접속 → 주인
        return Identity()

    name = _get(_HEADER_NAME)
    # authz 가 이름을 못 채웠으면(구글 subject id 등) 메일 로컬파트로 대체
    if not name or name.isdigit():
        name = email.split("@")[0]

    ident = Identity(
        email=email,
        name=name,
        role=_get(_HEADER_ROLE) or "guest",
        service=_get(_HEADER_SERVICE) or "mizuki",
        local=False,
    )
    logger.info(
        f"[identity] 접속자 {ident.label} <{ident.email}> "
        f"role={ident.role} owner={ident.is_owner}"
    )
    return ident
