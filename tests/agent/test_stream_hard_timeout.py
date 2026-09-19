"""Regression coverage for bounded non-terminating provider streams."""

import math
import threading
import time
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as helpers


class FakeAgent:
    def __init__(self, base_url="http://192.168.15.205:8000/v1"):
        self.base_url = base_url
        self.model = "dgx-spark"
        self.provider = "custom"
        self.api_mode = "chat_completions"
        self._interrupt_requested = False
        self.status = []
        self.activity = []

    def _buffer_status(self, value):
        self.status.append(value)

    def _touch_activity(self, value):
        self.activity.append(value)


def _call(agent=None):
    call = helpers._StreamingCall(
        agent or FakeAgent(),
        {"model": "dgx-spark", "messages": [{"role": "user", "content": "hi"}]},
        None,
    )
    call.clients.close_once = lambda _reason: None
    call._shutdown_stale_attempt_socket = lambda _response: None
    return call


def test_local_stream_hard_timeout_defaults_to_fifteen_minutes(monkeypatch):
    monkeypatch.delenv("HERMES_LOCAL_STREAM_HARD_TIMEOUT", raising=False)
    call = _call()
    call._resolve_hard_timeout()
    assert call._stream_hard_timeout == 900.0


def test_local_stream_hard_timeout_env_override_and_disable(monkeypatch):
    call = _call()
    monkeypatch.setenv("HERMES_LOCAL_STREAM_HARD_TIMEOUT", "37")
    call._resolve_hard_timeout()
    assert call._stream_hard_timeout == 37.0

    monkeypatch.setenv("HERMES_LOCAL_STREAM_HARD_TIMEOUT", "0")
    call._resolve_hard_timeout()
    assert math.isinf(call._stream_hard_timeout)


def test_remote_stream_has_no_new_default_hard_timeout(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_STREAM_HARD_TIMEOUT", "1")
    call = _call(FakeAgent(base_url="https://api.openai.com/v1"))
    call._resolve_hard_timeout()
    assert math.isinf(call._stream_hard_timeout)


def test_total_timeout_ignores_fresh_chunk_liveness():
    call = _call()
    call._stream_stale_timeout = 60.0
    call._stream_hard_timeout = 0.01
    call._stream_started_at = time.time() - 1.0
    call.last_chunk_time["t"] = time.time()
    call._call_done = threading.Event()
    call._monitor_interrupted = {"yes": False}

    call._monitor_loop()

    assert call._stream_hard_timeout_fired is True
    assert call._request_cancelled["value"] is True
    assert any("hard timeout" in item for item in call.agent.activity)


def test_run_surfaces_hard_timeout_without_stream_retry(monkeypatch):
    call = _call(FakeAgent(base_url="https://api.example.test/v1"))
    monkeypatch.setattr(helpers, "should_use_direct_api_call", lambda _agent: False)
    monkeypatch.setattr(call, "_resolve_stale_timeout", lambda: setattr(call, "_stream_stale_timeout", 60.0))
    monkeypatch.setattr(call, "_resolve_hard_timeout", lambda: setattr(call, "_stream_hard_timeout", 0.01))

    def slow_call():
        time.sleep(0.4)
        call._call_done.set()

    monkeypatch.setattr(call, "_run_call", slow_call)

    with pytest.raises(TimeoutError, match="hard wall-clock ceiling"):
        call.run()

    assert call._stream_hard_timeout_fired is True
