"""
solver/solver_log.py
v7.1.2 — Structured JSON logging for the entire solver pipeline.

Every meaningful event in the solver writes one JSON object per line to
`logs/solver.log`. The format is one-event-per-line so you can:

  - tail -f logs/solver.log               # live stream
  - grep '"event":"stage2_pick"' logs/solver.log
  - jq 'select(.fallback_used)' logs/solver.log
  - python scripts/tail_log.py            # pretty-printed live view

What we log (in order of importance):

  - LLM calls: every Stage 1 / Stage 2 / Stage 4 call gets a request log
    AND a response log. The response log includes elapsed time, response
    bytes, and any non-200 HTTP status (rate-limit 429s show up here).
  - ChromaDB queries: query text, top-k results with similarity scores,
    plus a count of how many landed via symbol vs semantic. This is how
    you verify "is my RAG actually working".
  - Stage 2 decisions: per-frontier-item, what candidates were shown,
    what the LLM picked, and what fallback_used (if any) was triggered.
  - Round transitions: round_num, frontier composition, what was
    resolved, what's still pending.
  - Errors: any exception with full traceback gets logged.

Why JSON-per-line and not a plain log:
  - Easy to filter by event type without grep-fu
  - Easy to count fallback events ('jq | wc -l')
  - Easy to extract latency distributions
  - Machine-friendly for future diagnostic tooling

Why a single file and not multiple:
  - One place to look. The whole point of this v7.1.2 work is "tell me
    what happened" — splitting across files makes that worse, not better.

How to enable/disable:
  - On by default. Set env var SOLVER_LOG=off to silence (file still
    created but only ERROR-level events written).
  - File location: ./logs/solver.log relative to project root.
"""
from __future__ import annotations
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "solver.log"


class JsonLineFormatter(logging.Formatter):
    """One JSON object per log line. Adds ISO timestamp + level + msg fields."""
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts":     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
                      + f".{int((record.created % 1) * 1000):03d}",
            "level":  record.levelname,
            "event":  record.msg if isinstance(record.msg, str) else "?",
        }
        # Anything attached via the `extra` kwarg shows up as record attributes.
        # Standard fields we want to skip:
        skip = {"name","msg","args","levelname","levelno","pathname","filename",
                "module","exc_info","exc_text","stack_info","lineno","funcName",
                "created","msecs","relativeCreated","thread","threadName",
                "processName","process","getMessage","message","taskName"}
        for k, v in record.__dict__.items():
            if k in skip:
                continue
            base[k] = v
        if record.exc_info:
            base["traceback"] = traceback.format_exception(*record.exc_info)
        try:
            return json.dumps(base, ensure_ascii=False, default=str)
        except Exception:
            # Last-ditch: stringify everything
            return json.dumps({"ts": base["ts"], "level": base["level"],
                               "event": base["event"],
                               "raw": str(base)})


_configured = False

def _configure_once() -> logging.Logger:
    global _configured
    logger = logging.getLogger("solver")
    if _configured:
        return logger
    logger.setLevel(logging.DEBUG if os.getenv("SOLVER_LOG_VERBOSE") else logging.INFO)
    logger.propagate = False  # don't double-emit via root logger
    LOG_DIR.mkdir(exist_ok=True)
    # Rotate at 10MB, keep last 5 files
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    if os.getenv("SOLVER_LOG", "on").lower() == "off":
        handler.setLevel(logging.ERROR)
    logger.addHandler(handler)
    # Also tee a one-time line so the user sees this file is alive
    logger.info("logger_initialized",
                extra={"log_file":     str(LOG_FILE),
                       "logger_level": logger.level,
                       "pid":          os.getpid()})
    _configured = True
    return logger


def log(event: str, **fields: Any) -> None:
    """
    The canonical way to log inside the solver. Usage:

        from solver.solver_log import log
        log("stage1_request", question=question, model=MODEL_FAST)

    Every call writes one JSON line. `event` is the event type (used for
    filtering). All other fields become structured data in the log line.

    Keep field values JSON-serializable; the formatter falls back to str()
    for anything weird, but you lose searchability.
    """
    logger = _configure_once()
    # Truncate very long string fields to keep log readable.
    # The whole point is observability, not full transcripts — if you need
    # the full LLM response, set SOLVER_LOG_VERBOSE=1.
    if not os.getenv("SOLVER_LOG_VERBOSE"):
        for k, v in list(fields.items()):
            if isinstance(v, str) and len(v) > 4000:
                fields[k] = v[:4000] + f"...[truncated, {len(v)} chars total]"
    logger.info(event, extra=fields)


def log_error(event: str, exc: BaseException | None = None, **fields: Any) -> None:
    """Log an error. If `exc` provided, traceback is included."""
    logger = _configure_once()
    if exc is not None:
        logger.error(event,
                     exc_info=(type(exc), exc, exc.__traceback__),
                     extra=fields)
    else:
        logger.error(event, extra=fields)
