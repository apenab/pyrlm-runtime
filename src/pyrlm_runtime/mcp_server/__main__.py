"""Command line entry point: ``python -m pyrlm_runtime.mcp_server`` or ``pyrlm-mcp``."""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from dotenv import load_dotenv

from pyrlm_runtime.mcp_server import (
    ConfigError,
    Secrets,
    ServerConfig,
    build_retriever,
    build_server,
    run_server,
)

logger = logging.getLogger("pyrlm_runtime.mcp_server")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyrlm-mcp",
        description="Serve pyrlm-runtime retrieval over MCP. Configure with PYRLM_MCP_* variables.",
    )
    parser.add_argument(
        "--env-file", help="load variables from this .env file before reading the environment"
    )
    args = parser.parse_args(argv)
    if args.env_file:
        load_dotenv(args.env_file, override=False)

    try:
        config = ServerConfig.from_env()
    except ConfigError as exc:
        print(f"pyrlm-mcp: invalid configuration: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(
        stream=sys.stderr, level=config.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    retriever = None
    try:
        secrets = Secrets.from_env()
        retriever = build_retriever(config, secrets)
        server = build_server(retriever=retriever, config=config)
        logger.info("serving over %s", config.transport)
        run_server(server, config, bearer_token=secrets.bearer_token)
    except ConfigError as exc:
        print(f"pyrlm-mcp: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
