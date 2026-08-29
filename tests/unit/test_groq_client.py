import json

import pytest

from benchmarks.baseline.groq_client import (
    GroqChatClient,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict,
    ) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    def read(self) -> bytes:
        return json.dumps(
            self.payload
        ).encode("utf-8")


def test_client_requires_api_key(
    monkeypatch,
):
    monkeypatch.delenv(
        "GROQ_API_KEY",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="GROQ_API_KEY is not set",
    ):
        GroqChatClient(
            model_name="example-model",
        )


def test_client_generates_and_records_metadata(
    monkeypatch,
):
    captured = {}

    def fake_urlopen(
        request,
        timeout,
    ):
        captured["request"] = request
        captured["timeout"] = timeout

        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                " Grounded answer. "
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "benchmarks.baseline.groq_client.urlopen",
        fake_urlopen,
    )

    client = GroqChatClient(
        model_name="test-model",
        api_key="test-secret",
        timeout_seconds=15,
    )

    result = client.generate(
        "Example prompt"
    )

    assert result.provider == "groq"
    assert result.model_name == "test-model"
    assert result.answer == "Grounded answer."
    assert result.latency_seconds >= 0

    body = json.loads(
        captured["request"].data.decode(
            "utf-8"
        )
    )

    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    assert (
        body["messages"][0]["content"]
        == "Example prompt"
    )

    headers = dict(
        captured["request"].header_items()
    )

    assert (
        headers["Accept"]
        == "application/json"
    )
    assert (
        headers["User-agent"]
        == "AI-Reliability-Platform/0.1"
    )
    assert captured["timeout"] == 15


def test_client_is_compatible_with_runner_callable(
    monkeypatch,
):
    monkeypatch.setattr(
        "benchmarks.baseline.groq_client.urlopen",
        lambda request, timeout: FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Answer"
                        }
                    }
                ]
            }
        ),
    )

    client = GroqChatClient(
        model_name="test-model",
        api_key="test-secret",
    )

    assert client("Prompt") == "Answer"
