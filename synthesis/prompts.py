"""Synthesis module: centralised prompt templates for all LLM-calling agents.

All raw prompt strings live here.  Agent modules import and format these
templates; no f-string prompts should appear inside business logic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core synthesis pipeline
# ---------------------------------------------------------------------------

ANALYZE_PAPERS_TEMPLATE = (
    "You are a research analyst. The user asks: '{query}'\n\n"
    "Below are excerpts from relevant research papers:\n\n{context}\n\n"
    "Identify and explain the key concepts, methods, and findings from these papers "
    "that are most relevant to the query. Be concise and technical."
)

SYNTHESIZE_FINDINGS_TEMPLATE = (
    "You are a research synthesis expert. The user asks: '{query}'\n\n"
    "Based on the following analysis of research papers:\n\n{analysis}\n\n"
    "Synthesize the key findings into a coherent narrative. "
    "Highlight agreements, contradictions, and research gaps. "
    "Conclude with the current state of the art."
)

GENERATE_IMPLEMENTATION_TEMPLATE = (
    "You are a senior AI systems engineer. The user asks: '{query}'\n\n"
    "Based on this research synthesis:\n\n{synthesis}\n\n"
    "Generate a concrete, production-ready implementation plan. Include:\n"
    "1. System architecture (modules, data flow)\n"
    "2. Technology stack and rationale\n"
    "3. Key algorithms or model choices\n"
    "4. Data pipeline design\n"
    "5. API design (endpoints, schemas)\n"
    "6. Testing strategy\n\n"
    "Be specific and actionable. Prefer Python, LangChain, FastAPI, and ChromaDB."
)

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

SCORE_CHUNKS_TEMPLATE = (
    "You are a research relevance scorer. Evaluate the following research excerpt "
    "on four dimensions (score each 0.0–1.0):\n"
    "- novelty: How new or groundbreaking is this research?\n"
    "- practicality: How applicable is this to real engineering problems?\n"
    "- adoption: How widely adopted or cited is this approach?\n"
    "- relevance: How relevant is this to the query '{query}'?\n\n"
    "Respond with a JSON object only, with keys: novelty, practicality, adoption, relevance.\n\n"
    "Research excerpt:\n{chunk}"
)

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

CLUSTER_CHUNKS_TEMPLATE = (
    "You are a research categorizer. Group the following research excerpts into "
    "thematic clusters.\n"
    "Return a JSON object mapping cluster names (short descriptive labels) to lists "
    "of excerpt indices (0-based integers).\n\n"
    "Excerpts:\n{excerpts}\n\n"
    "Respond with a JSON object only."
)

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

GENERATE_PROMPTS_TEMPLATE = (
    "You are a prompt engineer specialising in code generation.\n"
    "Based on the following implementation plan, generate 3–5 specific, actionable "
    "code-generation prompts that an AI coding assistant could use to implement "
    "individual components.\n\n"
    "Implementation plan:\n{implementation_plan}\n\n"
    "Return a JSON array of prompt strings only."
)

# ---------------------------------------------------------------------------
# Digest & feedback
# ---------------------------------------------------------------------------

DIGEST_TEMPLATE = (
    "You are a technical summariser producing a research digest.\n"
    "Summarise the following synthesis and implementation plan into a concise weekly "
    "digest suitable for a senior engineering team. Highlight the most impactful "
    "findings, key implementation decisions, and recommended next steps.\n\n"
    "Query: {query}\n\n"
    "Synthesis:\n{synthesis}\n\n"
    "Implementation plan:\n{implementation_plan}\n\n"
    "Produce a structured digest (max 400 words)."
)

FEEDBACK_TEMPLATE = (
    "You are a quality analyst reviewing a research synthesis pipeline output.\n"
    "Evaluate the following digest and suggest scoring weight adjustments "
    "(delta values between -0.2 and +0.2) for: novelty, practicality, adoption, relevance.\n"
    "Also rate the overall prompt quality (0.0–1.0) and suggest one improvement.\n\n"
    "Digest:\n{digest}\n\n"
    "Respond with a JSON object with keys:\n"
    "  weight_adjustments (object with novelty/practicality/adoption/relevance deltas),\n"
    "  prompt_quality (float),\n"
    "  prompt_improvement (string)."
)
