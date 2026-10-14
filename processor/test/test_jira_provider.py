from datetime import datetime, timezone

from processor.providers.jira import _jira_discourse_to_text
from processor.state_machine import DocumentRecord


def test_jira_discourse_to_text_renders_metadata_and_comments() -> None:
    doc = DocumentRecord(
        id="doc:jira:ENG/ENG-42",
        org_id="org:dev",
        source_provider="jira",
        source_scope="scope:jira:acme-atlassian-net:ENG",
        source_ref="ENG/ENG-42",
        title="ENG-42",
        s3_key="org:dev/jira/raw/ENG/ENG-42/v1.json",
        processing_state="raw",
        signal_score=None,
        char_count=None,
        current_version=1,
        source_created_at=datetime.now(timezone.utc),
    )
    payload = """
{
  "issue_key": "ENG-42",
  "title": "Move auth to service accounts",
  "issue_type": "Story",
  "status": "Done",
  "labels": ["eidetix-ingest", "auth"],
  "qualification_rule": "eligible_label_allowlist",
  "description": "We should switch to service tokens for automation.",
  "reporter": {"account_id": "a1", "display_name": "Alice"},
  "assignee": {"account_id": "b2", "display_name": "Bob"},
  "comments": [
    {
      "entry_id": "10",
      "created_at": "2026-08-22T10:00:00Z",
      "author": {"account_id": "c3", "display_name": "Charlie"},
      "body": "Decision: avoid user-scoped API tokens."
    }
  ]
}
""".strip()
    rendered = _jira_discourse_to_text(doc, payload)
    assert "# Move auth to service accounts" in rendered
    assert "Issue: ENG-42" in rendered
    assert "Labels: eidetix-ingest, auth" in rendered
    assert "Qualification rule: eligible_label_allowlist" in rendered
    assert "## Description" in rendered
    assert "## Discussion" in rendered
    assert "Decision: avoid user-scoped API tokens." in rendered
