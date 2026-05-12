"""Providers for default TASK user input values."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import yaml

from omnitool.gradio.config.task_template import TaskTemplate

logger = logging.getLogger(__name__)


class TaskUserValueProvider(Protocol):
    """Lookup source for default values used to prefill TASK user inputs."""

    def values_for_template(
        self,
        template: TaskTemplate,
        template_path: str | None = None,
    ) -> dict[str, Any]: ...


class EmptyTaskUserValueProvider:
    """Provider used when no default source is configured."""

    def values_for_template(
        self,
        template: TaskTemplate,
        template_path: str | None = None,
    ) -> dict[str, Any]:
        return {}


class FileTaskUserValueProvider:
    """Read TASK user input defaults from a YAML or JSON file.

    Supported shape:

    templates:
      Template Name:
        user_field: value
      template-file-stem:
        user_field: value
      template.yaml:
        user_field: value
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def values_for_template(
        self,
        template: TaskTemplate,
        template_path: str | None = None,
    ) -> dict[str, Any]:
        data = self._load()
        raw_templates = data.get("templates", data)
        if not isinstance(raw_templates, dict):
            logger.warning("%s: task user values must be a mapping.", self.path)
            return {}

        for key in self._candidate_keys(template, template_path):
            values = raw_templates.get(key)
            if isinstance(values, dict):
                return dict(values)
            if values is not None:
                logger.warning(
                    "%s: values for template %r must be a mapping.",
                    self.path,
                    key,
                )
        return {}

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                if self.path.suffix.lower() == ".json":
                    data = json.load(fh)
                else:
                    data = yaml.safe_load(fh) or {}
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            logger.warning("Failed to load task user values from %s: %s", self.path, exc)
            return {}
        if not isinstance(data, dict):
            logger.warning("%s: task user values file must contain a mapping.", self.path)
            return {}
        return data

    @staticmethod
    def _candidate_keys(
        template: TaskTemplate,
        template_path: str | None,
    ) -> list[str]:
        keys: list[str] = []
        if template.name:
            keys.append(template.name)
        if template_path:
            path = Path(template_path)
            keys.extend([path.name, path.stem, str(path)])
        return list(dict.fromkeys(keys))


def build_task_user_value_provider(path: str | None) -> TaskUserValueProvider:
    if path:
        return FileTaskUserValueProvider(path)
    return EmptyTaskUserValueProvider()
