---
id: 002
title: Create comprehensive implementation plan
stage: plan
date: 2025-12-06
surface: agent
model: claude-haiku-4-5-20251001
feature: textbook-spec
branch: 001-textbook-spec
user: user
command: /sp.plan
labels: ["architecture","phases","research","quality","testing"]
links:
  spec: specs/001-textbook-spec/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-textbook-spec/plan.md
tests:
  - Constitution alignment checks (passed all 6 principles)
  - Architecture validation (decoupled Docusaurus + FastAPI)
---

## Prompt

Create a comprehensive implementation plan that strictly adheres to the Constitution and Specification, organized into the following mandatory sections: Architecture Sketch, Section Structure, Research Approach, Quality Validation, Decisions Needing Documentation, and Testing Strategy.

Implementation Phases (Organizational Structure):
The Plan MUST break down all tasks by the user's defined phases:
- Phase A: Research (Deep dive into ROS 2, Isaac, VLA concepts using Claude Code search/context).
- Phase B: Foundation (Setup Docusaurus, FastAPI, Qdrant/Neon infrastructure, and Better-Auth scaffolding).
- Phase C: Analysis (Content generation for all 4 Modules, RAG integration, and user profiling implementation).
- Phase D: Synthesis (Personalization, Translation, Subagent extraction, and final quality checks).

Section Content Details:
- Architecture Sketch: Draw the system diagram showing the decoupled flow (Docusaurus F/E -> FastAPI B/E -> Qdrant/Neon D/B).
- Section Structure: Map the book's FOUR MODULES (ROS 2, Digital Twin, AI-Robot Brain, VLA) across the four project phases (A-D).
- Research Approach: Detail the research-concurrent writing process, ensuring content generation happens alongside the technical setup, strictly adhering to the Constitution's APA citation standard and rigor.
- Quality Validation: Define validation checks based on the Constitution's Rigor/Accuracy (e.g., code snippets must run in a ROS 2/Python environment, content must be technically accurate).
- Decisions Needing Documentation: List and define key technical choices, options, and tradeoffs (e.g., choice of embedding model, Better-Auth integration method, serverless deployment strategy).
- Testing Strategy: Detail validation checks (based on the Specification's success criteria) for the RAG Chatbot, Content Personalization feature, and Better-Auth user profiling flow.

## Response snapshot

Created comprehensive implementation plan (plan.md) adhering to Constitution v1.0.0 with:

**Architecture Sketch**: Decoupled system with Docusaurus frontend (React/Node.js) + FastAPI backend (Python 3.11+) communicating via REST API. Backend services: Auth (Better-Auth + Neon), Content (lesson serving), Chatbot (RAG pipeline with Qdrant + LLM), Personalization, Translation. Databases: Neon PostgreSQL (users, content, progress) + Qdrant vectors (embeddings).

**Constitution Alignment**: All 6 principles PASS - Rigor & Accuracy, Academic Clarity, Reproducibility, Four-Module Format, Decoupled Architecture, Reusability Mandate.

**Project Structure**: Frontend (Docusaurus, React, TypeScript) + Backend (FastAPI, Python) + Databases (Neon + Qdrant) + CI/CD. Documentation includes spec, plan, research, data-model, contracts, quickstart.

**Implementation Phases**:
- Phase A (Research, Weeks 1-2): ROS 2 APIs, peer-reviewed sources, citation database (APA format)
- Phase B (Foundation, Weeks 2-4): Docusaurus + FastAPI + Neon + Qdrant + Better-Auth + OpenAPI contracts
- Phase C (Analysis, Weeks 4-8): 4 modules × 30+ lessons = 120+ total; ROS 2 → Digital Twin → AI-Robot Brain → VLA
- Phase D (Synthesis, Weeks 8-10): Personalization, translation, integration testing, quality validation

**Research Approach**: Content generation parallel with technical setup. Every lesson cites official docs/peer-reviewed sources (APA). Code examples tested in ROS 2 Humble. Citation database + accuracy tracking matrix.

**Quality Validation**: 4 gates - Content Accuracy (API verification, citations, reproducibility), Code Testing (pytest suite), Pedagogical Review (clarity, progression, accessibility), Performance (load testing, uptime).

**Decisions Needing Documentation** (ADRs):
1. Embedding Model (OpenAI vs. HuggingFace vs. Anthropic)
2. LLM Provider (GPT-4 vs. Claude vs. self-hosted)
3. Auth Strategy (Better-Auth vs. custom)
4. Translation Approach (human vs. machine vs. hybrid)
5. Qdrant Deployment (self-hosted vs. managed cloud)

**Testing Strategy**: 30 test cases across 5 user stories. SC-001 to SC-030 validation matrix maps each success criterion to test method, owner, and phase. Test tools: pytest (backend), Vitest (frontend), Playwright (E2E), load testing for performance.

## Outcome

- ✅ Impact: Complete implementation roadmap with clear phases, architectural decisions, quality gates, and testing strategy. All Constitution principles enforced.
- 🧪 Tests: Constitution alignment checklist (6/6 pass), architecture validation (decoupled confirmed), phase breakdown (A-D sequenced)
- 📁 Files: specs/001-textbook-spec/plan.md (comprehensive 20KB+ document with all required sections)
- 🔁 Next prompts: `/sp.tasks` (generate actionable tasks from plan), `/sp.adr` (document embedding model, LLM, auth decisions)
- 🧠 Reflection: Plan successfully translates specification + Constitution into implementable roadmap. Phased approach enables parallel content generation and technical development. Quality gates ensure academic rigor throughout.

## Evaluation notes (flywheel)

- Failure modes observed: None; all Constitution gates passed, architecture validated
- Graders run and results (PASS/FAIL): PASS (6/6 Constitution principles aligned, all phases clearly defined, testing strategy comprehensive)
- Prompt variant (if applicable): Initial plan for MVP (P1 features: content, chatbot, auth; P2-P3 deferred to later)
- Next experiment (smallest change to try): Resolve ADRs (especially embedding model and LLM choice) before Phase B foundation work begins
