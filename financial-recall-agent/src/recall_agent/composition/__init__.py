"""Explicit composition root for approved in-process rule templates."""

from src.recall_agent.core.plugins import PluginRegistry
from src.recall_agent.templates.reward_missing.plugin import RewardMissingPlugin


PLUGIN_REGISTRY = PluginRegistry([RewardMissingPlugin()])
