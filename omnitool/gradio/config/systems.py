"""Per-system configuration registry, keyed by ``TaskExecution.system``.

Mirrors the :data:`PLATFORM_PROMPTS` registry pattern. Each entry declares
whether the system runs in a browser (enables DevTools-based table extraction),
optional login config, and a system-specific prompt fragment appended to the
agent's system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LoginConfig:
    """Login flow definition for a system.

    Fields are natural-language targets resolvable by the GTA1 grounding model
    or text patterns matched against on-screen content.
    """

    url: str
    username_field: str
    password_field: str
    submit_target: str
    success_pattern: str


@dataclass(frozen=True)
class SystemConfig:
    """Configuration for one target system referenced by ``TaskExecution.system``.

    Attributes:
        name: System identifier as it appears in the YAML template.
        is_browser: True when the agent operates this system through a web browser.
            Enables DevTools-based table extraction in PR 3.
        prompt_fragment: System-specific instructions appended to the agent
            system prompt. Empty string when no extra guidance is needed.
        login: Optional login flow descriptor. Consumed by a future login runner.
    """

    name: str
    is_browser: bool
    prompt_fragment: str = ""
    login: Optional[LoginConfig] = None


SYSTEMS: Dict[str, SystemConfig] = {
    "EPA": SystemConfig(
        name="EPA",
        is_browser=True,
        prompt_fragment="",
    ),
}


def get_system_config(name: str) -> Optional[SystemConfig]:
    """Return the :class:`SystemConfig` for *name*, or ``None`` if unknown."""
    return SYSTEMS.get(name)
