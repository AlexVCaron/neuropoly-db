import os
from typing import Optional

import httpx
from fastapi import FastAPI, Header, Response, status
from cachetools import TTLCache

GITHUB_API = "https://api.github.com"
GITHUB_ORG = os.getenv("NB_GATEWAY_GITHUB_ORG", "")
CACHE_TTL = int(os.getenv("GITHUB_VALIDATION_CACHE_TTL", "60"))
CACHE_MAX = int(os.getenv("GITHUB_VALIDATION_CACHE_MAX", "1024"))

cache = TTLCache(maxsize=CACHE_MAX, ttl=CACHE_TTL)
app = FastAPI()


async def validate_token(token: str) -> Optional[str]:
    # Return username if token valid and user is in org, else None
    if token in cache:
        return cache[token]

    headers = {"Authorization": token, "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{GITHUB_API}/user", headers=headers)
        except Exception:
            return None
        if r.status_code != 200:
            return None
        user = r.json().get("login")
        if not user:
            return None

        # Check org membership by listing user's orgs
        try:
            r2 = await client.get(f"{GITHUB_API}/user/orgs", headers=headers)
        except Exception:
            return None
        if r2.status_code != 200:
            return None
        orgs = [o.get("login") for o in r2.json() if o.get("login")]
        if GITHUB_ORG and GITHUB_ORG not in orgs:
            return None

        cache[token] = user
        return user


@app.get("/validate")
@app.head("/validate")
async def validate(authorization: Optional[str] = Header(None)):
    if not authorization:
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)
    username = await validate_token(authorization)
    if not username:
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)
    headers = {"X-Auth-Request-User": username}
    return Response(status_code=status.HTTP_200_OK, headers=headers)

# also accept any path to be flexible for proxying
@app.get("/{path:path}")
@app.head("/{path:path}")
async def validate_any(path: str, authorization: Optional[str] = Header(None)):
    return await validate(authorization)
