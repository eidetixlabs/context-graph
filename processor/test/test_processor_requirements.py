from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_processor_requirements_pin_mcp_v1_for_pydantic_ai() -> None:
    requirements = (ROOT / "processor" / "requirements.txt").read_text()
    assert "pydantic-ai[mcp]>=1.90.0,<2.0.0" in requirements
    assert "mcp>=1.9,<2" in requirements


def test_processor_dockerfile_smoke_checks_pydantic_ai_mcp_import() -> None:
    dockerfile = (ROOT / "processor" / "Dockerfile").read_text()
    assert "from pydantic_ai.mcp import MCPServerStreamableHTTP" in dockerfile
