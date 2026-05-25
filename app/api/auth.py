from __future__ import annotations

import time
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from hashlib import sha256

import httpx
from fastapi import Header, HTTPException, Request, status

from app.rag.config import Settings, get_settings


@dataclass(frozen=True)
class ApiKeyContext:
    api_key_id: str
    organization_id: str
    permissions: tuple[str, ...]


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization bearer token.",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization must use Bearer authentication.",
        )
    return token.strip()


def _base64_urlsafe_no_padding(value: bytes) -> str:
    return urlsafe_b64encode(value).decode("ascii").rstrip("=")


def validate_mechbase_api_key(value: str, settings: Settings | None = None) -> ApiKeyContext:
    settings = settings or get_settings()
    if not settings.convex_http_url or not settings.convex_recording_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Convex API key validation is not configured.",
        )

    key_hash = sha256(value.encode("utf-8")).digest()
    key_hash_value = _base64_urlsafe_no_padding(key_hash)
    url = settings.convex_http_url.rstrip("/") + "/validateApiKey"

    try:
        response = httpx.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-convex-recording-secret": settings.convex_recording_secret,
            },
            json={"keyHash": key_hash_value},
            timeout=10,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to validate API key.",
        ) from exc

    if response.status_code == status.HTTP_401_UNAUTHORIZED:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to validate API key.",
        )

    payload = response.json()
    if not payload.get("valid"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    return ApiKeyContext(
        api_key_id=payload["apiKeyId"],
        organization_id=payload["workspaceId"],
        permissions=tuple(payload.get("permissions") or ()),
    )


def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
) -> ApiKeyContext:
    settings = get_settings()
    token = _extract_bearer_token(authorization)
    context = validate_mechbase_api_key(token, settings)

    required_permission = settings.required_api_key_permission
    if required_permission and required_permission not in context.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key must include the {required_permission!r} permission.",
        )

    request.state.api_key_context = context
    return context


def record_usage(
    *,
    context: ApiKeyContext,
    request: Request,
    status_code: int,
    started_at: float,
    settings: Settings | None = None,
) -> None:
    settings = settings or get_settings()
    if not settings.convex_http_url or not settings.convex_recording_secret:
        return

    url = settings.convex_http_url.rstrip("/") + "/recordUsage"
    payload = {
        "workspaceId": context.organization_id,
        "apiKeyId": context.api_key_id,
        "endpoint": request.url.path,
        "method": request.method,
        "statusCode": status_code,
        "timestamp": int(time.time() * 1000),
        "latencyMs": int((time.perf_counter() - started_at) * 1000),
        "requestId": request.headers.get("x-request-id"),
        "userAgent": request.headers.get("user-agent"),
    }

    try:
        httpx.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-convex-recording-secret": settings.convex_recording_secret,
            },
            json=payload,
            timeout=5,
        )
    except httpx.HTTPError:
        # Usage analytics should never break retrieval.
        return
