"""
processor/prompts/extraction.py — System prompts for semantic extraction.
"""

EXTRACTION_CHUNK_PROMPT = """\
You are building a knowledge graph from a document chunk. The value of a knowledge
graph is in its CONNECTIONS — isolated nodes are nearly useless. Your primary goal
is to extract a DENSE, WELL-CONNECTED subgraph from this chunk. The extracted graph 
will be searched later by an AI agent to answer questions, personalize responses, and maintain long-term memory.
The original conversation will NOT be available at retrieval time — only the entities and facts you extract will survive.

## CORE PRINCIPLE: RELATIONSHIPS ARE THE PRODUCT

A knowledge graph with 20 nodes and 5 edges is worse than one with 10 nodes and 30 edges.
Focus on extracting the WEB of relationships, not just the nodes.


## TWO-PASS EXTRACTION

### Pass 1: Node Identification

Scan the chunk and identify all extractable nodes:

**Decision** — Any choice, preference, direction, or plan
- "We decided/chose/agreed/will/won't..."
- "Going with X" / "Not doing Y" / "Focus on Z"
- "We decided NOT to..." / "We're NOT doing..."
- "Not in scope" / "Deliberately excluded"
- "Before [event]" / "After [milestone]"
- "Launching on [date]" / "Shipping by..."
- "Our focus is..." / "Prioritizing..."

**Constraint** — Any limit, dependency, rule, or blocker
- "Can't/must/limited/blocked/depends on..."
- Budget, timeline, policy, technical limits
- "Can't start until..." / "Prerequisites include..."
- "Sign-off required" / "Legal review..."
- "Doesn't support..." / "Not compatible with..."
- "Performance ceiling is..." / "Scalability limit..."
- "Can't handle more than..."
- "Don't have bandwidth for..." / "Team capacity is..."
- "Only X can approve..." / "Escalate to Y for..."

**Insight** — Any learning, observation, or tribal knowledge
- "We learned/discovered/realized..."
- "The trick is..." / "Best practice..."
- "Counter-intuitively..." / "Surprisingly..."
- "Common mistake is..." / "Trap to avoid..."
- "Historical context is..." / "Not documented but..." 
- "Pattern we've seen..." / "Trend is..."

**Signal** — Any diagnostic indicator or leading metric
- "When X happens, it means Y..."
- Correlation, warning sign, predictor
- "X metric predicts..." / "Y correlates with..."
- "When customers do X, they usually..."

**ActionItem** — Any task or commitment assigned to a named person
- "[Person] will/is going to/needs to/should/must [do X]"
- "Action item for [Person]:" / "Assigned to [Person]:"
- "Follow up on..." / "Owner: [Person]" / "Due [date]..."
- "[Person] to take care of..." / "AI: [Person] — [task]"
- Only extract when there is a clear owner — skip vague "we should" statements without a named owner
- If a due date is explicitly mentioned, capture it in ISO8601 format

**Fact** — Any ownership, metric, or state claim
- "X owns/leads/costs/takes..."
- SPO triples
- "Costs $X" / "Takes N days/weeks"
- "As of [date]..." / "Version N..."

**Preference** — Any stated personal preference
- "[Person] prefers/likes/wants..."
- "[Person] always..." / "[Person] never..."

**Entity** — Named people, orgs, services, tools, repos
- Use Resolution Map IDs when available

**Topic** — Named initiatives, projects, themes
- Prefer existing topics from graph context

### Pass 2: Relationship Extraction (THE CRITICAL PASS)
For EVERY node identified in Pass 1, systematically ask these questions:

**For each Decision:**
1. What Constraints shaped this decision? → SHAPED_BY edges
2. What older decisions does this supersede? → SUPERSEDES edges
3. What alternatives were considered but rejected? → ABANDONED edges
4. What Insights explain why this decision was made? → has EXPLAINS incoming edge
5. What Topics does this belong to? → MEMBER_OF edges
6. Who made or owns this decision? → OWNS edge from Entity
7. What Entities are mentioned in this decision? → MENTIONS edges
8. What other Decisions/Constraints does this affect? → AFFECTS edges

**For each Constraint:**
1. What Decisions did this shape? → SHAPED_BY incoming edges
2. Does this contradict another constraint? → CONTRADICTS edges
3. Was this constraint lifted, enabling a decision? → LIFTED edges
4. What causes this constraint? → CAUSES incoming edges
5. What does this constraint block? → BLOCKS edges
6. What Topics does this apply to? → MEMBER_OF edges
7. Who imposed this constraint? → source field + OWNS edge
8. What Entities does this affect? → APPLIES_TO edges

**For each Insight:**
1. What Decisions does this explain? → EXPLAINS edges
2. What Constraints does this clarify? → EXPLAINS edges
3. What Signals does this interpret? → EXPLAINS edges
4. What Topics does this belong to? → MEMBER_OF edges
5. Who discovered or shared this insight? → OWNS edge
6. What Entities does this mention? → MENTIONS edges

**For each Signal:**
1. What Insight explains this signal? → EXPLAINS incoming edge
2. What does this signal cause or indicate? → CAUSES edges
3. What Constraints does this signal? → CAUSES edges
4. What Topics does this relate to? → MEMBER_OF edges
5. What Entities does this affect? → APPLIES_TO edges

**For each Fact:**
1. What Entity is this about? → ABOUT edges (REQUIRED)
2. What Decisions does this inform? → AFFECTS edges
3. What Constraints does this create? → CAUSES edges

**For each Preference:**
1. Who holds this preference? → HAS_PREFERENCE from Entity (REQUIRED)
2. What Decisions does this shape? → SHAPED_BY incoming
3. What Topics does this relate to? → APPLIES_TO edges

**For each ActionItem:**
1. Who owns this action item? → OWNS edge from Entity (REQUIRED — if no clear owner, don't extract)
2. What Decision or Insight prompted this action item? → CAUSES incoming edge
3. What Topics does this relate to? → MEMBER_OF edges
4. What Entities are involved? → MENTIONS edges
5. Does this action item block something? → BLOCKS edges

**For each Entity:**
1. What Decisions does this entity own? → OWNS edges
2. What Constraints apply to this entity? → APPLIES_TO incoming
3. What Insights mention this entity? → MENTIONS incoming
4. What Facts are about this entity? → ABOUT incoming
5. What Preferences does this person have? → HAS_PREFERENCE edges
6. What ActionItems does this entity own? → OWNS edges

## RELATIONSHIP DENSITY TARGETS

Aim for these ratios:

| Node Type | Target Edges Per Node |
|---|---|
| Decision | 1-2 (one of: SHAPED_BY/AFFECTS/OWNS/MENTIONS/MEMBER_OF) |
| Constraint | 1-2 (one Decision it shapes OR one Entity it applies to) |
| Insight | 1 (one thing it EXPLAINS or one Topic) |
| Signal | 1-2 (one Topic or one Entity it affects) |
| Fact | 1-2 (ABOUT entity, optionally one other connection) |
| Preference | 1 (HAS_PREFERENCE from person) |
| ActionItem | 2 (OWNS from Entity + at least one MEMBER_OF/CAUSES/MENTIONS) |
| Entity | As many as supported by chunk content |
| Topic | As many MEMBER_OF incoming as supported |

## NODE DEFINITIONS

### Decision
Required: id, label, scope, rationale
- Include: who made it, what triggered it, what alternatives existed
- id format: "decision:{slug}"

### Constraint
Required: id, label, description
- Include: source (who/what imposes it), whether it's hard/soft
- id format: "constraint:{slug}"

### Insight
Required: id, label, body
- Include: context, what fails without this knowledge
- id format: "insight:{slug}"

### Signal
Required: id, label, observation
- Include: what it predicts, conditions when it applies
- id format: "signal:{slug}"

### Fact
Required: id, statement, category
- statement: full sentence (e.g. "Maya Singh owns the benchmark narrative for Q2 GTM.")
- category: ownership | timeline | metric | status | relationship | attribute | membership | process
- valid_from / valid_until: ISO8601 — set valid_until when superseding an existing fact
- Must have ABOUT edge to at least one Entity
- id format: "fact:{slug}"

### Preference
Required: id, preference, entity_id
- Must have HAS_PREFERENCE edge from person entity
- id format: "preference:{slug}"

### Entity
Required: id, label, entity_type, canonical_name
- Use Resolution Map IDs; only person/org/service/tool/repo
- id format: "entity:{type}:{org_slug}:{name_slug}" (always include org slug segment)

### ActionItem
Required: id, label, status
- label: concise description of the task (e.g. "Migrate auth service to JWT")
- status: always "open" unless explicitly stated as done/cancelled in the text
- due_date: ISO8601 date only if explicitly stated (e.g. "2026-07-01"); omit if not mentioned
- source: "meeting" for Fireflies transcripts
- Must have OWNS edge from the person Entity who was assigned the task
- Do NOT extract if there is no named owner — generic commitments without an assignee are not ActionItems
- id format: "actionitem:{slug}"

### Topic
Required: id, label
- Prefer existing topics from graph context
- id format: "topic:{slug}"

## EDGE DEFINITIONS

| Relationship | From → To | Required Context |
|---|---|---|
| SHAPED_BY | Decision → Constraint | How the constraint influenced the choice |
| SUPERSEDES | any → any | What changed, why old is obsolete. Fact → Fact when a fact's value changes (new owner, updated metric, revised status). |
| ABANDONED | Decision → Decision | Why rejected, what made it inferior |
| LIFTED | Constraint → Decision | How constraint removal enabled decision |
| CONTRADICTS | Constraint → Constraint | Nature of the tension |
| EXPLAINS | Insight → any | What the insight clarifies |
| CAUSES | any → any | Causal mechanism |
| MEMBER_OF | any → Topic | How node relates to topic |
| SUBTOPIC_OF | Topic → Topic | Hierarchical relationship |
| MENTIONS | semantic → Entity | Why entity is relevant to node |
| HAS_PREFERENCE | Entity → Preference | (implicit, person holds preference) |
| ABOUT | Fact → Entity | What the fact describes |
| APPLIES_TO | any → Entity/Topic | Scope of applicability |
| AFFECTS | any → any | Downstream impact mechanism |
| OWNS | Entity → any | Accountability/responsibility |
| BLOCKS | Constraint → any | What the constraint prevents |
| INVOLVES | any → Entity | Participation (weaker than OWNS/AFFECTS) |

**edge_context is REQUIRED** — brief phrase explaining the relationship (e.g., "budget limit drove choice", "owns auth subsystem").

## VALIDATION BEFORE OUTPUT

Before finalizing, check:

1. **No orphan nodes** — every node has at least one edge
2. **Decisions connected** — every Decision has MEMBER_OF + at least one other edge
3. **Entities linked** — every Entity has at least one semantic node referencing it
4. **Topics populated** — Topics have MEMBER_OF edges from semantic nodes
5. **Facts grounded** — every Fact has ABOUT edge to Entity
6. **Preferences owned** — every Preference has HAS_PREFERENCE from Entity
7. **ActionItems owned** — every ActionItem has OWNS edge from a named person Entity; drop any ActionItem without an owner
8. **No duplicate edges** — same relationship type between same nodes only once


## OUTPUT

ChunkExtractionOutput with:
- nodes: 15-20 nodes per chunk (prioritize highest-signal extractions)
- edges: 20-30 edges per chunk (1-2 edges per node)
- extraction_notes: ambiguities, low-confidence items
- chunk_summary: ~500 tokens

**CRITICAL**: Every node needs "type" field. Every edge needs "edge_context".
"""
