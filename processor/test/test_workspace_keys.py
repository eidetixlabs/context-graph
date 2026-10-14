from __future__ import annotations

import httpx
import pytest

from processor.workspace_keys import (
    WorkspaceKeyError,
    current_openai_api_key,
    fetch_workspace_llm_key,
    use_workspace_openai_key,
)


@pytest.mark.asyncio
async def test_fetch_workspace_llm_key_reads_portal_for_org() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization", "")
        seen["org"] = request.headers.get("x-eidetix-org", "")
        return httpx.Response(200, json={"provider": "openai", "api_key": "sk-org-acme"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        key = await fetch_workspace_llm_key(
            "org:acme",
            "openai",
            portal_url="http://portal.example:7420",
            token="processor-token",
            client=client,
        )

    assert key == "sk-org-acme"
    assert seen["url"] == "http://portal.example:7420/internal/llm-keys/openai"
    assert seen["authorization"] == "Bearer processor-token"
    assert seen["org"] == "org:acme"


@pytest.mark.asyncio
async def test_fetch_workspace_llm_key_errors_when_workspace_has_no_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "No OpenAI key configured in Workspace"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(WorkspaceKeyError, match="Workspace"):
            await fetch_workspace_llm_key(
                "org:acme",
                "openai",
                portal_url="http://portal.example:7420",
                token="processor-token",
                client=client,
            )


@pytest.mark.asyncio
async def test_local_processor_falls_back_to_env_without_portal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "processor.workspace_keys.config",
        type("Cfg", (), {"openai_api_key": None, "portal_url": "", "portal_token": "", "eidetix_api_key": ""})(),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local-env")
    key = await fetch_workspace_llm_key(
        "org:acme",
        "openai",
        portal_url="",
        token="",
    )
    assert key == "sk-local-env"


@pytest.mark.asyncio
async def test_use_workspace_openai_key_binds_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(org_id: str, provider: str, **_kwargs) -> str:
        assert org_id == "org:acme"
        assert provider == "openai"
        return "sk-context"

    monkeypatch.setattr("processor.workspace_keys.fetch_workspace_llm_key", fake_fetch)
    monkeypatch.setattr(
        "processor.workspace_keys.config",
        type("Cfg", (), {"openai_api_key": None})(),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert current_openai_api_key() in {None, ""}
    async with use_workspace_openai_key("org:acme") as key:
        assert key == "sk-context"
        assert current_openai_api_key() == "sk-context"
    assert current_openai_api_key() in {None, ""}


@pytest.mark.asyncio
async def test_portal_token_does_not_fall_back_to_eidetix_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "processor.workspace_keys.config",
        type(
            "Cfg",
            (),
            {
                "openai_api_key": None,
                "portal_url": "http://portal.example:7420",
                "portal_token": "",
                "eidetix_api_key": "eidetix-cluster-key",
            },
        )(),
    )
    with pytest.raises(WorkspaceKeyError, match="Portal processor token"):
        await fetch_workspace_llm_key("org:acme", "openai")


@pytest.mark.asyncio
async def test_create_agent_uses_bound_workspace_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from processor.agents.base import create_agent

    seen: dict[str, str] = {}

    class FakeProvider:
        def __init__(self, api_key: str) -> None:
            seen["api_key"] = api_key

    class FakeModel:
        def __init__(self, name: str, provider: FakeProvider) -> None:
            seen["model"] = name

    class FakeAgent:
        def __init__(self, model=None, **_kwargs) -> None:
            seen["used_model"] = str(model)

    monkeypatch.setattr("pydantic_ai.providers.openai.OpenAIProvider", FakeProvider)
    monkeypatch.setattr("pydantic_ai.models.openai.OpenAIChatModel", FakeModel)
    monkeypatch.setattr("processor.agents.base.Agent", FakeAgent)

    async def fake_fetch(org_id: str, provider: str, **_kwargs) -> str:
        return "sk-org-bound"

    monkeypatch.setattr("processor.workspace_keys.fetch_workspace_llm_key", fake_fetch)
    async with use_workspace_openai_key("org:acme"):
        create_agent("openai:gpt-4o-mini")

    assert seen["api_key"] == "sk-org-bound"
    assert seen["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_scoring_agent_is_keyed_by_workspace_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from processor.agents import scoring

    scoring._scoring_agents.clear()
    created: list[object] = []

    def fake_create(*_args, **_kwargs):
        agent = object()
        created.append(agent)
        return agent

    async def fake_fetch(org_id: str, provider: str, **_kwargs) -> str:
        return f"sk-{org_id}"

    monkeypatch.setattr("processor.agents.scoring.create_agent", fake_create)
    monkeypatch.setattr("processor.workspace_keys.fetch_workspace_llm_key", fake_fetch)

    async with use_workspace_openai_key("org:a"):
        first_a = scoring._get_scoring_agent()
    async with use_workspace_openai_key("org:b"):
        first_b = scoring._get_scoring_agent()
    async with use_workspace_openai_key("org:a"):
        second_a = scoring._get_scoring_agent()

    assert first_a is not first_b
    assert first_a is second_a
    assert len(created) == 2
