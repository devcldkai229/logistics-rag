"""
Log luồng MACA ra terminal (stderr) — các khối quan sát orchestrator → … → judge.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from typing import Any

_logger = logging.getLogger("maca.demo")
_stream_ready = False


def _ensure_terminal_handler() -> None:
    global _stream_ready
    if _stream_ready:
        return
    _stream_ready = True
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(h)


def log_agent_flow(agent: str, title: str, **fields: Any) -> None:
    """Ghi một khối log dễ đọc trên terminal hiện tại."""
    _ensure_terminal_handler()
    lines: list[str] = [f"━━ MACA │ {agent} │ {title} ━━"]
    for key, val in fields.items():
        if val is None:
            continue
        if isinstance(val, str) and (len(val) > 180 or "\n" in val):
            lines.append(f"  ▸ {key}:")
            lines.append(textwrap.indent(val.rstrip(), "     "))
        elif isinstance(val, dict):
            lines.append(f"  ▸ {key}: {val!r}")
        else:
            lines.append(f"  ▸ {key}: {val!r}")
    _logger.info("\n".join(lines))
