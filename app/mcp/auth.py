from __future__ import annotations

import logging

import httpx
from mcp.server.auth.provider import AccessToken
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class _ClerkActiveToken(BaseModel):
    id: str
    client_id: str
    subject: str
    scopes: list[str]
    revoked: bool
    expired: bool
    expiration: float | None


class ClerkTokenVerifier:
    """Validate opaque or JWT OAuth access tokens through Clerk's Backend API."""

    def __init__(
        self,
        *,
        secret_key: str | None,
        issuer_url: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._secret_key = secret_key
        self._issuer_url = issuer_url.rstrip("/")
        self._http_client = http_client

    async def verify_token(self, token: str) -> AccessToken | None:
        if not self._secret_key:
            return None

        try:
            response = await self._post_token(token)
        except httpx.HTTPError as exc:
            logger.warning("Clerk OAuth token verification failed: %s", type(exc).__name__)
            return None

        if response.status_code != 200:
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("Clerk returned a non-JSON OAuth token verification response")
            return None
        if not isinstance(payload, dict):
            logger.warning("Clerk returned an unexpected OAuth token verification response")
            return None
        if payload.get("active") is False:
            return None

        try:
            verified = _ClerkActiveToken.model_validate(payload)
        except ValidationError:
            logger.warning("Clerk returned an unexpected OAuth token verification response")
            return None

        if verified.revoked or verified.expired:
            return None

        return AccessToken(
            token=token,
            client_id=verified.client_id,
            scopes=verified.scopes,
            expires_at=int(verified.expiration) if verified.expiration is not None else None,
            subject=verified.subject,
            claims={"iss": self._issuer_url, "token_id": verified.id},
        )

    async def _post_token(self, token: str) -> httpx.Response:
        request = {
            "method": "POST",
            "url": "https://api.clerk.com/v1/oauth_applications/access_tokens/verify",
            "headers": {"Authorization": f"Bearer {self._secret_key}"},
            "json": {"access_token": token},
            "timeout": 10,
        }
        if self._http_client is not None:
            return await self._http_client.request(**request)
        async with httpx.AsyncClient() as client:
            return await client.request(**request)
