"""Small explicit registry for approved rule-template plugins."""

from __future__ import annotations

from collections.abc import Iterable

from src.recall_agent.core.models import TemplatePlugin


class PluginRegistry:
    def __init__(self, plugins: Iterable[TemplatePlugin]) -> None:
        self._plugins = {plugin.rule_template: plugin for plugin in plugins}

    def require(self, rule_template: str) -> TemplatePlugin:
        try:
            return self._plugins[rule_template]
        except KeyError as exc:
            raise ValueError(
                f"No plugin registered for rule_template={rule_template!r}. "
                f"Available: {sorted(self._plugins)}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))
