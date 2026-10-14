// Entity nodes are global directory records. Private provenance belongs on
// scoped semantic nodes and relationships, never on the shared Entity node.
MATCH (e:Entity)
REMOVE e.source_scope, e.source_doc_id, e.source_s3_url, e.source_chat, e.ingested_by;
