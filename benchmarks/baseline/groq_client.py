from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GROQ_CHAT_COMPLETIONS_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)


@dataclass(frozen=True)
class GroqGeneration:
    provider: str
    model_name: str
    answer: str
    latency_seconds: float


class GroqChatClient:
    """Minimal Groq client for experimental Base-RAG runs."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        clean_model_name = model_name.strip()

        if not clean_model_name:
            raise ValueError(
                "model_name must not be empty"
            )

        resolved_api_key = (
            api_key
            or os.environ.get("GROQ_API_KEY")
            or ""
        ).strip()

        if not resolved_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive"
            )

        self.model_name = clean_model_name
        self._api_key = resolved_api_key
        self.timeout_seconds = float(
            timeout_seconds
        )

    def generate(
        self,
        prompt: str,
    ) -> GroqGeneration:
        clean_prompt = prompt.strip()

        if not clean_prompt:
            raise ValueError(
                "prompt must not be empty"
            )

        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": clean_prompt,
                    }
                ],
                "temperature": 0,
            }
        ).encode("utf-8")

        request = Request(
            GROQ_CHAT_COMPLETIONS_URL,
            data=payload,
            headers={
                "Authorization": (
                    f"Bearer {self._api_key}"
                ),
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": (
                    "AI-Reliability-Platform/0.1"
                ),
            },
            method="POST",
        )

        started_at = time.perf_counter()

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_response = (
                    response.read().decode("utf-8")
                )

        except HTTPError as exc:
            raise RuntimeError(
                "Groq API request failed "
                f"with HTTP {exc.code}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                "Groq API request failed"
            ) from exc

        latency_seconds = (
            time.perf_counter() - started_at
        )

        try:
            body = json.loads(raw_response)
            answer = (
                body["choices"][0]["message"]["content"]
                .strip()
            )
        except (
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "Groq returned an invalid response"
            ) from exc

        if not answer:
            raise RuntimeError(
                "Groq returned an empty answer"
            )

        return GroqGeneration(
            provider="groq",
            model_name=self.model_name,
            answer=answer,
            latency_seconds=round(
                latency_seconds,
                4,
            ),
        )

    def __call__(
        self,
        prompt: str,
    ) -> str:
        return self.generate(prompt).answer
