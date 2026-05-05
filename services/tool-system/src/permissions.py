"""Permission system: role-based access control for tool execution."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time

logger = logging.getLogger(__name__)

# Role hierarchy: admin > operator > agent > readonly
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"*"},
    "operator": {"shell", "filesystem", "http_get", "code_execution", "search", "database"},
    "agent": {"http_get", "search", "filesystem_read", "code_execution"},
    "readonly": {"search", "filesystem_read"},
}

# Tool category → minimum required role
TOOL_ROLE_REQUIREMENTS: dict[str, str] = {
    "shell": "operator",
    "filesystem": "operator",
    "filesystem_read": "agent",
    "http_get": "agent",
    "http_post": "operator",
    "code_execution": "agent",
    "search": "agent",
    "database": "operator",
    "system": "admin",
}


class PermissionManager:
    def __init__(self, secret_key: str) -> None:
        self._secret = secret_key.encode()

    def can_execute(self, role: str, tool_name: str) -> bool:
        allowed = ROLE_PERMISSIONS.get(role, set())
        if "*" in allowed:
            return True
        if tool_name in allowed:
            return True

        # Check category-level permission
        required_role = TOOL_ROLE_REQUIREMENTS.get(tool_name, "agent")
        role_levels = ["readonly", "agent", "operator", "admin"]
        try:
            role_idx = role_levels.index(role)
            req_idx = role_levels.index(required_role)
            return role_idx >= req_idx
        except ValueError:
            return False

    def generate_token(self, role: str, ttl_seconds: int = 3600) -> str:
        payload = json.dumps({"role": role, "exp": int(time.time()) + ttl_seconds})
        sig = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        data = base64.b64encode(f"{payload}|{sig}".encode()).decode()
        return data

    def verify_token(self, token: str) -> str:
        try:
            decoded = base64.b64decode(token.encode()).decode()
            payload_str, sig = decoded.rsplit("|", 1)
            expected_sig = hmac.new(self._secret, payload_str.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                logger.warning("Token signature mismatch")
                return "readonly"
            payload = json.loads(payload_str)
            if payload.get("exp", 0) < time.time():
                logger.warning("Token expired")
                return "readonly"
            return payload.get("role", "agent")
        except Exception:
            return "agent"  # default for internal service calls
