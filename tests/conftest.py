import socket

import pytest


@pytest.fixture(autouse=True)
def no_paid_requests(monkeypatch):
    monkeypatch.setenv("SURPLUS_API_KEY", "test-only")

    def deny_network(*args, **kwargs):
        raise AssertionError("Tests must use mocked HTTP, never real network requests")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
