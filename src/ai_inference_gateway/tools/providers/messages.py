"""Prompt -> chat transcript conversion shared by all provider adapters.

A prompt whose first line starts with a role marker (``System:``, ``User:``,
``Human:``, ``Assistant:``, ``AI:``) is treated as a transcript: system turns
become the system prompt, the rest become strictly alternating user/assistant
messages (consecutive same-role turns are merged) starting with a user turn.
Anything else, including malformed transcripts, is sent verbatim as a single
user message.
"""

import re
from typing import Final, Literal

from pydantic import Field

from ai_inference_gateway.schemas import StrictModel

ChatRole = Literal["user", "assistant"]

_ROLE_MARKER_RE: Final = re.compile(
    r"^[ \t]*(system|user|human|assistant|ai)[ \t]*:", re.IGNORECASE | re.MULTILINE
)
_ROLE_ALIASES: Final[dict[str, ChatRole | Literal["system"]]] = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
}


class ChatMessage(StrictModel):
    """One conversational turn."""

    role: ChatRole
    content: str = Field(min_length=1)


class ChatTranscript(StrictModel):
    """System prompt plus alternating messages, always starting with ``user``."""

    system: str | None = None
    messages: tuple[ChatMessage, ...] = Field(min_length=1)


def to_transcript(prompt: str) -> ChatTranscript:
    """Convert ``prompt`` into a provider-neutral transcript."""
    single = ChatTranscript(messages=(ChatMessage(role="user", content=prompt),))
    markers = list(_ROLE_MARKER_RE.finditer(prompt))
    if not markers or prompt[: markers[0].start()].strip():
        return single

    system_parts: list[str] = []
    turns: list[ChatMessage] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(prompt)
        content = prompt[marker.end() : end].strip()
        if not content:
            continue
        role = _ROLE_ALIASES[marker.group(1).lower()]
        if role == "system":
            system_parts.append(content)
        elif turns and turns[-1].role == role:
            turns[-1] = ChatMessage(role=role, content=f"{turns[-1].content}\n\n{content}")
        else:
            turns.append(ChatMessage(role=role, content=content))

    if not turns or turns[0].role != "user":
        return single
    return ChatTranscript(
        system="\n\n".join(system_parts) or None,
        messages=tuple(turns),
    )
