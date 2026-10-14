"""Lazy public processor-worker exports.

Importing a lightweight provider (such as github_profile markdown rendering)
must not load the full worker/Neo4j stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NotionProcessorWorker",
    "GithubMarkdownProcessorWorker",
    "GithubDiscourseProcessorWorker",
    "JiraProcessorWorker",
]


def __getattr__(name: str) -> Any:
    if name == "NotionProcessorWorker":
        from processor.providers.notion import NotionProcessorWorker

        return NotionProcessorWorker
    if name == "GithubMarkdownProcessorWorker":
        from processor.providers.github_markdown import GithubMarkdownProcessorWorker

        return GithubMarkdownProcessorWorker
    if name == "GithubDiscourseProcessorWorker":
        from processor.providers.github import GithubDiscourseProcessorWorker

        return GithubDiscourseProcessorWorker
    if name == "JiraProcessorWorker":
        from processor.providers.jira import JiraProcessorWorker

        return JiraProcessorWorker
    raise AttributeError(name)
