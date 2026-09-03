"""Install a meta-path hook that patches `app.api.utility.request_data`.

This script is safe to `import` (it installs the hook at import time)
and can also be executed directly. It merges a local JSON vocabulary file
with the upstream imaging_modalities response when present.
"""

import sys
import json
import sys
from pathlib import Path

_VOCAB_PATH = Path("/usr/src/neurobagel/neuropoly_imaging_modalities.json")


class _NeuropolyVocabFinder:
    """A simple meta-path finder/loader that patches `app.api.utility` on import."""

    def find_module(self, name, path=None):
        if name == "app.api.utility":
            return self
        return None

    def load_module(self, name):
        # If already loaded, return it.
        if name in sys.modules:
            return sys.modules[name]

        # Remove ourselves to avoid recursion while importing the real module.
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass

        import importlib

        mod = importlib.import_module(name)

        # Patch the request_data function to merge local vocab when available.
        try:
            _orig = getattr(mod, "request_data")

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
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"[neuropoly-vocab] patch error: {exc}", file=sys.stderr, flush=True)
                return _orig(url, err)

            setattr(mod, "request_data", _patched)
        except Exception:  # pragma: no cover - if module shape unexpected, just skip patch
            pass

        sys.modules[name] = mod
        return mod


# Install the hook (idempotent)
if not any(isinstance(p, _NeuropolyVocabFinder) for p in sys.meta_path):
    sys.meta_path.insert(0, _NeuropolyVocabFinder())


def main():
    """Entrypoint for direct execution: report status and exit.

    The installation runs at import time; running the script prints the
    status of the local vocab file and exits with code 0.
    """
    if _VOCAB_PATH.exists():
        print(f"[neuropoly-vocab] local vocab present: {_VOCAB_PATH}")
    else:
        print(f"[neuropoly-vocab] local vocab not found: {_VOCAB_PATH}")


if __name__ == "__main__":
    main()
