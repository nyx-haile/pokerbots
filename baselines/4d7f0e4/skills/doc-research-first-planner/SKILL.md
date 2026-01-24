---
name: doc-research-first-planner
description: Enforce evidence-first planning and Deep Research gating. Use for any request involving planning, architecture/design, implementation steps, non-trivial modifications, integrations, migrations, refactors, project-specific behavior questions, or version-sensitive/best-practice claims. Requires consulting design docs then docs/; if task is not simple per gate, output a Deep Research request and stop.
---

# Doc Research First Planner

## Overview
Enforce a mandatory evidence chain before planning or implementation by consulting local design docs, vendor docs, and (when required) Deep Research reports.

## Workflow (Evidence-First)

1. **(Optional) Index docs for fast lookup.**
   - Run `python3 skills/doc-research-first-planner/scripts/index_docs.py` from repo root to build `.cache/design_index.json` and `.cache/docs_index.json`.
2. **Consult design docs first (in order).**
   - Search: `design/**`, `docs/design/**`, `adr/**` or `adrs/**`, `architecture/**`, `specs/**`, `rfcs/**`.
   - Use `README*` / `CONTRIBUTING*` only if relevant.
   - Allow missing paths.
3. **Consult vendor/product docs mirror.**
   - Search `docs/**` (exclude `docs/design/**` which belongs to design docs).
4. **Apply the Complexity Gate.**
   - If any criterion fails or is uncertain, treat as **Not Simple**.
5. **Produce Evidence Digest (always).**
   - Use `templates/evidence_digest.md`.
   - Include file paths and capture constraints/invariants and requirements.
6. **If Not Simple -> output Deep Research Request and stop.**
   - Use `templates/deep_research_request.md`.
   - Do not provide plan or code.
7. **If Simple -> provide plan (and optional implementation).**
   - Provide plan, test plan, rollback plan.
   - Implement only if explicitly requested by the user.
8. **After Deep Research report exists in `./research/**`.**
   - Read the report(s), add a **Research Digest** with file paths.
   - Then plan/implement grounded in design + docs + research.

## Complexity Gate (Deterministic)

Treat as **Simple** only if ALL are true:
- Changes limited to **<= 2 files** OR a single localized module.
- No new dependency or external integration.
- No schema/data migration.
- No auth/security model changes.
- No API contract changes.
- No architecture decision needed (no new patterns).
- Design/docs clearly cover required behavior.

Default to **Not Simple** if uncertain.

## Conflict Rule
Treat design docs as authoritative for project behavior when conflicts appear. Note the conflict in the Evidence Digest.

## Stop Assuming Contract
- Do not claim "best/latest/current" without local docs or Deep Research.
- Do not invent project conventions; confirm in design docs or repo.
- If docs are absent/contradictory/unclear, request Deep Research rather than guessing.
- List any remaining assumptions explicitly (ideally none).

## Templates
- `templates/evidence_digest.md` for **Design Digest** + **Docs Digest**.
- `templates/deep_research_request.md` for the exact Deep Research request format.
- `templates/plan.md` for plan/test/rollback structure after digests.

## Scripts
- `scripts/index_docs.py` builds `.cache/design_index.json` and `.cache/docs_index.json` for faster discovery.
