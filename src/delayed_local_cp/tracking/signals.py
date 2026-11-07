"""Graceful stop: SIGTERM/SIGINT set a flag that the run loop checks each round."""

from __future__ import annotations

import signal

_STOP = {"requested": False}


def _handler(signum, frame):  # noqa: ARG001
    _STOP["requested"] = True


def install() -> None:
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def stop_requested() -> bool:
    return _STOP["requested"]


def reset() -> None:
    _STOP["requested"] = False


def request_stop() -> None:
    """Used by tests to simulate a shutdown."""
    _STOP["requested"] = True
