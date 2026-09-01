# Auto-installed sitecustomize to inject neuropoly vocab and API auth middleware.
import sys
import json
from pathlib import Path
import os

_VOCAB_PATH = Path("/usr/src/neurobagel/neuropoly_imaging_modalities.json")

# Ensure graph credentials are present for app startup (development convenience).
try:
    if not os.environ.get("NB_GRAPH_USERNAME"):
        os.environ["NB_GRAPH_USERNAME"] = "dbuser"
    if not os.environ.get("NB_GRAPH_PASSWORD"):
        p = Path("/tmp/db_user_password_file")
        if p.exists() and p.is_file():
            os.environ["NB_GRAPH_PASSWORD"] = p.read_text().strip()
        else:
            # Fallback to environment or leave unset
            pass
except Exception:
    pass

class _NeuropolyVocabFinder:
    def find_module(self, name, path=None):
        if name == "app.api.utility":
            return self
        return None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        sys.meta_path.remove(self)
        import importlib
        mod = importlib.import_module(name)
        _orig = mod.request_data
        def _patched(url, err):
            if "imaging_modalities.json" in url and _VOCAB_PATH.exists():
                try:
                    base = _orig(url, err)
                    custom = json.loads(_VOCAB_PATH.read_text())
                    if isinstance(base, dict):
                        base = [base]
                    result = base + custom
                    print(
                        f"[neuropoly-vocab] patch applied: {len(result)} namespace blocks, "
                        f"{sum(len(ns.get('terms', [])) for ns in result)} total terms",
                        file=sys.stderr,
                        flush=True,
                    )
                    return result
                except Exception as exc:
                    print(f"[neuropoly-vocab] patch error: {exc}", file=sys.stderr, flush=True)
            return _orig(url, err)
        mod.request_data = _patched
        sys.modules[name] = mod
        return mod

sys.meta_path.insert(0, _NeuropolyVocabFinder())

# API auth middleware finder
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response
    import httpx
except Exception:
    BaseHTTPMiddleware = None

class _ApiAuthMiddlewareFinder:
    def find_module(self, name, path=None):
        if name == "app.main":
            return self
        return None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        sys.meta_path.remove(self)
        import importlib
        mod = importlib.import_module(name)
        try:
            app = getattr(mod, "app", None)
            if app is not None and BaseHTTPMiddleware is not None:
                class _GatewayAuthMiddleware(BaseHTTPMiddleware):
                    async def dispatch(self, request, call_next):
                        auth = request.headers.get("authorization")
                        if auth and auth.lower().startswith("bearer"):
                            try:
                                async with httpx.AsyncClient(timeout=10.0) as client:
                                    resp = await client.get(
                                        "http://token_validator:4181/validate",
                                        headers={"Authorization": auth},
                                    )
                                if resp.status_code == 200:
                                    user = resp.headers.get("X-Auth-Request-User")
                                    if user:
                                        hdrs = list(request.scope.get("headers", []))
                                        hdrs.append((b"x-auth-user", user.encode("utf-8")))
                                        request.scope["headers"] = hdrs
                                        return await call_next(request)
                                return Response(status_code=401)
                            except Exception:
                                return Response(status_code=502)
                        return await call_next(request)
                app.add_middleware(_GatewayAuthMiddleware)
        except Exception:
            pass
        sys.modules[name] = mod
        return mod

sys.meta_path.insert(0, _ApiAuthMiddlewareFinder())
