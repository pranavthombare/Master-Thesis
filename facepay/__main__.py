"""``uv run facepay`` - serve the API and frontend."""

import argparse
import logging

import uvicorn

from .api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="facepay", description="Serve FacePay (API + web UI).")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (camera access needs localhost or HTTPS)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
