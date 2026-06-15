"""Claude 도구 사용 승인 중계기.

letta가 띄운 `claude -p` 서브프로세스의 PreToolUse hook이 보내는 권한 요청을
프론트(VTuber 웹)로 중계하고, 네무린이 승인/거부할 때까지 기다린 뒤 결정을 돌려준다.

흐름:
    claude hook ──HTTP POST /tool-approval──▶ ToolApprovalManager.request_approval()
        → 연결된 프론트로 WS "tool-approval-request" 전송
        → tool-approval-response 가 올 때까지 future 대기 (timeout)
        → 결정(JSON)을 hook에게 반환

allowlist("항상 승인")의 영구 저장/TTL 만료 판정은 hook 쪽이 담당한다.
이 매니저는 실시간 1회 승인/거부만 중계한다.
"""

import asyncio
import json
from typing import Dict
from uuid import uuid4

from fastapi import WebSocket
from loguru import logger

# 네무린이 응답하지 않을 때 거부로 떨어지는 시간(초).
# hook 쪽 HTTP 타임아웃과 claude hook timeout 보다 짧게 둔다.
# 3겹 레이어링: 백엔드 300 < hook HTTP 310 < claude hook 330
DEFAULT_APPROVAL_TIMEOUT = 300.0


class ToolApprovalManager:
    """tool_use_id(request_id)별로 승인 future를 관리하는 중계기."""

    def __init__(self) -> None:
        self._pending: Dict[str, asyncio.Future] = {}

    async def request_approval(
        self,
        clients: Dict[str, WebSocket],
        payload: dict,
        timeout: float = DEFAULT_APPROVAL_TIMEOUT,
    ) -> dict:
        """프론트로 승인 요청을 보내고 결정이 올 때까지 대기.

        Args:
            clients: client_uid -> WebSocket (연결된 프론트들)
            payload: hook이 보낸 도구 정보 (tool_name, tool_input, rule_label 등)
            timeout: 응답 대기 한계(초)

        Returns:
            결정 dict. 최소한 {"decision": "allow"|"deny"} 를 담는다.
            - 승인:     {"decision": "allow", "scope": "once"|"always", ...}
            - 거부:     {"decision": "deny", "reason": "..."}
            - 무응답/미연결: {"decision": "deny", "reason": "timeout"|"no_client"}
        """
        if not clients:
            logger.warning("[ToolApproval] 연결된 프론트가 없어 자동 거부")
            return {"decision": "deny", "reason": "no_client"}

        request_id = uuid4().hex
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future

        message = json.dumps(
            {
                "type": "tool-approval-request",
                "request_id": request_id,
                "tool_name": payload.get("tool_name", ""),
                "tool_input": payload.get("tool_input", {}),
                # '항상 승인' 시 고를 수 있는 범위 후보 [{id, label}]
                "rule_options": payload.get("rule_options", []),
                "ttl_days": payload.get("ttl_days", 1),
                # 미즈키가 이 도구를 쓰려는 이유 (reasoning)
                "intent": payload.get("intent", ""),
            },
            ensure_ascii=False,
        )

        # 연결된 모든 프론트로 브로드캐스트 — 누구든 먼저 응답하면 채택.
        sent = 0
        for uid, ws in list(clients.items()):
            try:
                await ws.send_text(message)
                sent += 1
            except Exception as e:
                logger.warning(f"[ToolApproval] 클라이언트 {uid} 전송 실패: {e}")
        if sent == 0:
            self._pending.pop(request_id, None)
            return {"decision": "deny", "reason": "no_client"}

        logger.info(
            f"[ToolApproval] 승인 요청 전송 (tool={payload.get('tool_name')}, "
            f"request_id={request_id[:8]}, clients={sent})"
        )

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[ToolApproval] {timeout}초 무응답 — 거부 처리 (request_id={request_id[:8]})")
            return {"decision": "deny", "reason": "timeout"}
        finally:
            self._pending.pop(request_id, None)

        logger.info(f"[ToolApproval] 결정 수신: {result.get('decision')} (request_id={request_id[:8]})")
        return result

    def resolve(self, request_id: str, result: dict) -> None:
        """프론트가 보낸 tool-approval-response 로 대기 중인 future를 깨운다."""
        future = self._pending.get(request_id)
        if future is None:
            logger.warning(f"[ToolApproval] 알 수 없거나 만료된 request_id: {request_id[:8] if request_id else request_id}")
            return
        if not future.done():
            future.set_result(result)
