from base.tools import query as q


def test_sanitize_fulltext_query_splits_terms_with_or() -> None:
    assert q._sanitize_fulltext_query("mcp timeout keep") == "mcp OR timeout OR keep"


def test_sanitize_fulltext_query_escapes_lucene_chars() -> None:
    assert q._sanitize_fulltext_query("service/api") == "service\\/api"


def test_agent_node_view_uses_type_content_and_source_doc() -> None:
    node = {
        "id": "decision:foo",
        "label": "Use streamable transport",
        "rationale": "Avoid SSE framing edge-cases in tool responses.",
        "trigger": "Intermittent parse errors",
        "source_doc_id": "doc:notion:123",
        "embedding": [0.1, 0.2],
    }
    out = q._agent_node_view("decision:foo", node, "Decision")
    assert out["id"] == "decision:foo"
    assert out["type"] == "Decision"
    assert out["label"] == "Use streamable transport"
    assert out["content"] == "Avoid SSE framing edge-cases in tool responses."
    assert out["source_doc"] == "doc:notion:123"
    assert "embedding" not in out
