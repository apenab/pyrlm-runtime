"""Shared Azure + model connectivity preflight check for example scripts."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _load_from_zshrc() -> None:
    """Parse ~/.zshrc and inject any missing Azure vars into os.environ."""
    zshrc = Path.home() / ".zshrc"
    if not zshrc.is_file():
        return

    pattern = re.compile(
        r"""^\s*export\s+(AZURE_OPENAI_API_KEY|OPENAI_ENDPOINT|AZURE_ACCOUNT_NAME|AZURE_OPENAI_API_VERSION)\s*=\s*(['"]?)(.+?)\2\s*$"""
    )
    try:
        for line in zshrc.read_text(encoding="utf-8").splitlines():
            m = pattern.match(line)
            if m:
                key, _, value = m.group(1), m.group(2), m.group(3)
                if not os.environ.get(key):
                    os.environ[key] = value
    except Exception:
        pass


def check_azure_connection(model: str, api_version: str | None = None) -> None:
    """Verify Azure env vars are set and the model responds.

    Resolution order for env vars:
      1. Current process environment (already set in shell)
      2. ~/.zshrc export statements (fallback for uv run / non-login shells)

    Exits with a clear error message if the check fails.
    """
    # --- 1. Load missing vars from ~/.zshrc ---
    _load_from_zshrc()

    # --- 2. Validate required vars ---
    missing: list[str] = []
    if not os.environ.get("AZURE_OPENAI_API_KEY", "").strip():
        missing.append("AZURE_OPENAI_API_KEY")
    if not os.environ.get("OPENAI_ENDPOINT", "").strip() and \
       not os.environ.get("AZURE_ACCOUNT_NAME", "").strip():
        missing.append("OPENAI_ENDPOINT (or AZURE_ACCOUNT_NAME)")

    if missing:
        print("ERROR: Missing required Azure environment variable(s):")
        for v in missing:
            print(f"  - {v}")
        print("\nEnsure these are exported in ~/.zshrc or set in the shell before running.")
        sys.exit(1)

    # --- 3. Show what will be used ---
    from pyrlm_runtime.adapters import AzureOpenAIAdapter

    endpoint_env = os.environ.get("OPENAI_ENDPOINT") or os.environ.get("AZURE_ACCOUNT_NAME", "")
    key_preview = os.environ.get("AZURE_OPENAI_API_KEY", "")[:6] + "..."

    kwargs: dict = {"model": model, "timeout": 30.0}
    if api_version:
        kwargs["api_version"] = api_version

    # Build adapter just to read the resolved endpoint URL
    try:
        adapter = AzureOpenAIAdapter(**kwargs)
    except EnvironmentError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"  env endpoint : {endpoint_env}")
    print(f"  resolved URL : {adapter.endpoint}")
    print(f"  api key      : {key_preview}")
    print(f"  model        : {model}")

    # Debug: show env vars after loading from zshrc
    import socket
    try:
        # Pre-resolve hostname to catch DNS issues early
        hostname = adapter.endpoint.split("//")[1].split("/")[0]
        addr = socket.gethostbyname(hostname)
        print(f"  hostname     : {hostname} → {addr}")
    except socket.gaierror as e:
        print(f"  hostname DNS : FAILED ({e})")
        sys.exit(1)

    print(f"Checking connection ...", end=" ", flush=True)

    # --- 4. Minimal API call ---
    try:
        import httpx
        resp = adapter.complete(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=100,  # GPT-5.1 needs headroom before emitting content
            temperature=0.0,
        )
        adapter.close()
        answer = (resp.text or "").strip().lower()
        if not answer:
            print("FAILED (empty response — model returned no content)")
            sys.exit(1)
        print(f"OK  ✓  (response: {answer!r})\n")
    except httpx.HTTPStatusError as exc:
        print(f"FAILED")
        print(f"  HTTP {exc.response.status_code}: {exc.response.text[:300]}")
        sys.exit(1)
    except Exception as exc:
        print(f"FAILED")
        print(f"  {type(exc).__name__}: {exc}")
        sys.exit(1)
