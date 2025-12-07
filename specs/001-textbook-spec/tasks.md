---

description: "Task list for AI-Native Physical AI & Humanoid Robotics Textbook implementation"

---

# Tasks: AI-Native Physical AI & Humanoid Robotics Textbook

**Input**: Design documents from `/specs/001-textbook-spec/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)

**Tests**: Tests are NOT included in this task list (not explicitly requested in specification). Implementation focuses on feature delivery.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions

- **Frontend**: `frontend/src/`
- **Backend**: `backend/src/`
- **Database**: `backend/src/db/`
- **Tests**: `backend/tests/` and `frontend/tests/`

---

## Phase 1: Setup (Project Initialization & Infrastructure)

**Purpose**: Initialize projects, configure dependencies, setup development environment

- [x] T001 Create project root directory structure with `frontend/` and `backend/` folders
- [x] T002 [P] Initialize Node.js project in frontend folder with package.json for Docusaurus 3.0+, React 18+, Tailwind CSS
- [x] T003 [P] Initialize Python project in backend folder with pyproject.toml, requirements.txt for FastAPI, Pydantic, SQLAlchemy, pytest
- [x] T004 [P] Configure linting and formatting tools: ESLint + Prettier (frontend), Black + Ruff (backend)
- [x] T005 [P] Setup GitHub Actions CI/CD pipeline in `.github/workflows/ci.yaml` for lint, test, build
- [x] T006 Create `.env.example` file documenting all required environment variables (Neon URL, Qdrant endpoint, LLM API keys, Better-Auth config)
- [x] T007 Setup git hooks (.husky) for pre-commit checks (linting, formatting validation)

---

## Phase 2: Foundational (Blocking Prerequisites for ALL User Stories)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Database Setup

- [ ] T008 Create Neon PostgreSQL migration scripts in `backend/src/db/alembic/versions/` for initial schema
- [ ] T009 [P] Define SQLAlchemy ORM models in `backend/src/models/`:
  - [ ] T009a [P] `backend/src/models/user.py` (User entity: user_id, email, password_hash, ros_experience_level, focus_area, language_preference)
  - [ ] T009b [P] `backend/src/models/module.py` (Module entity: module_id, name enum, description, order_index)
  - [ ] T009c [P] `backend/src/models/lesson.py` (Lesson entity: lesson_id, module_id FK, title, learning_objectives, content_markdown, code_examples, diagrams, order_index)
  - [ ] T009d [P] `backend/src/models/user_progress.py` (UserProgress entity: progress_id, user_id FK, lesson_id FK, completed_at, bookmarked, time_spent_seconds)
  - [ ] T009e [P] `backend/src/models/chatbot_query.py` (ChatbotQuery entity: query_id, user_id FK, query_text, retrieved_passages, response_text, response_generation_time_ms)
  - [ ] T009f [P] `backend/src/models/lesson_embedding.py` (LessonEmbedding entity: embedding_id, lesson_id FK, passage_text, embedding_vector)
  - [ ] T009g [P] `backend/src/models/content_translation.py` (ContentTranslation entity: translation_id, lesson_id FK, language_code, translated_title, translated_content_markdown, reviewed_at)
- [ ] T010 Configure database connection in `backend/src/db/session.py` with SQLAlchemy engine, session factory, dependency injection for FastAPI

### FastAPI Backend Scaffolding

- [ ] T011 Create FastAPI application entry point in `backend/src/main.py` with:
  - CORS middleware configuration (allow frontend origin)
  - Global exception handlers
  - Health check endpoint (`GET /health`)
- [ ] T012 Create environment configuration module in `backend/src/config.py` with pydantic Settings for database URL, Qdrant endpoint, LLM API keys, authentication secrets
- [ ] T013 [P] Setup Pydantic schemas in `backend/src/schemas/`:
  - [ ] T013a [P] `backend/src/schemas/user.py` (UserCreate, UserUpdate, UserResponse, UserProfile)
  - [ ] T013b [P] `backend/src/schemas/lesson.py` (LessonCreate, LessonResponse, ModuleResponse)
  - [ ] T013c [P] `backend/src/schemas/chatbot.py` (ChatbotQuery request/response, CitationResponse)
  - [ ] T013d [P] `backend/src/schemas/personalization.py` (RecommendationRequest, RecommendationResponse)
  - [ ] T013e [P] `backend/src/schemas/auth.py` (SignupRequest, ProfileRequest, AuthResponse)
- [ ] T014 Create dependency injection utilities in `backend/src/dependencies.py` for database sessions, user authentication verification, rate limiting

### Authentication & Authorization

- [ ] T015 Integrate Better-Auth library in `backend/src/services/auth_service.py` with:
  - Email/password signup and login flows
  - Session token generation and validation
  - Password reset flow
  - Rate limiting on signup (max 5 accounts per IP per hour)
- [ ] T016 Create authentication middleware in `backend/src/middleware/auth_middleware.py` for:
  - Token verification on protected routes
  - Session state management
  - User context injection into requests

### Vector Database Setup

- [ ] T017 Initialize Qdrant client in `backend/src/services/qdrant_client.py` with:
  - Connection to Qdrant endpoint (self-hosted or managed cloud)
  - Collection creation for lesson embeddings (dimension 1536 for OpenAI embeddings)
  - Index configuration for vector search

### LLM Integration

- [ ] T018 Create LLM integration in `backend/src/utils/embeddings.py` with:
  - Embedding generation function (OpenAI text-embedding-3-large or HuggingFace all-MiniLM-L6-v2)
  - LLM response generation function (OpenAI GPT-4 or Claude API)
  - Prompt templates for RAG context
  - Citation extraction logic

### Frontend Setup

- [ ] T019 Initialize Docusaurus configuration in `frontend/docusaurus.config.ts` with:
  - Site metadata and structure
  - Sidebar configuration for 4 modules
  - Theme customization with Tailwind CSS
  - Static site generation output
- [ ] T020 Create frontend API client in `frontend/src/services/api.client.ts` with:
  - Axios or fetch-based HTTP client
  - Error handling and retry logic
  - Base URL configuration from environment
  - Request/response interceptors for authentication

### OpenAPI Contracts

- [ ] T021 [P] Create OpenAPI specification files in `backend/specs/contracts/`:
  - [ ] T021a [P] `backend/specs/contracts/auth-api.yaml` (signup, login, profile GET/PATCH, logout endpoints)
  - [ ] T021b [P] `backend/specs/contracts/content-api.yaml` (GET lesson, GET module, GET module roadmap, POST lesson view)
  - [ ] T021c [P] `backend/specs/contracts/chatbot-api.yaml` (POST query, GET response, GET citations)
  - [ ] T021d [P] `backend/specs/contracts/personalization-api.yaml` (GET recommendations, POST engagement tracking)
  - [ ] T021e [P] `backend/specs/contracts/translation-api.yaml` (GET translated lesson, PATCH language preference, GET supported languages)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Access and Learn from Structured Textbook Content (Priority: P1) 🎯 MVP

**Goal**: Deliver four modules with 120+ lessons accessible via Docusaurus frontend, code examples executable without modification

**Independent Test**: Student can navigate all 4 modules, view lessons, code examples execute successfully, page loads in ≤3 seconds

### Implementation for User Story 1

- [ ] T022 [P] [US1] Create Docusaurus module pages for all 4 modules in `frontend/src/pages/`:
  - [ ] T022a [P] [US1] Create ROS 2 Fundamentals module landing page with module overview, lesson list, prerequisites
  - [ ] T022b [P] [US1] Create Digital Twin & Simulation module landing page
  - [ ] T022c [P] [US1] Create AI-Robot Brain module landing page
  - [ ] T022d [P] [US1] Create Vision Language Action (VLA) module landing page
- [ ] T023 [P] [US1] Create 30+ lesson markdown files for ROS 2 Fundamentals in `frontend/src/pages/ROS2Fundamentals/`:
  - Each lesson must include: title, learning objectives, explanatory content, code examples (copy-paste ready), diagrams, cross-references
  - Code examples must cite official ROS 2 Humble documentation
  - All claims must cite sources (peer-reviewed or official docs)
- [ ] T024 [P] [US1] Create 30+ lesson markdown files for Digital Twin & Simulation in `frontend/src/pages/DigitalTwin/`
- [ ] T025 [P] [US1] Create 30+ lesson markdown files for AI-Robot Brain in `frontend/src/pages/AIRobotBrain/`
- [ ] T026 [P] [US1] Create 30+ lesson markdown files for Vision Language Action in `frontend/src/pages/VLA/`
- [ ] T027 [US1] Implement LessonViewer React component in `frontend/src/components/LessonViewer/` with:
  - Syntax-highlighted code examples
  - Diagram rendering
  - Previous/Next lesson navigation
  - Learning objectives display
- [ ] T028 [US1] Implement ModuleRoadmap React component in `frontend/src/components/ModuleRoadmap/` with:
  - Visual lesson progression display
  - Current/completed/locked lesson indicators (for authenticated users)
  - Lesson filtering by module
  - Link to lessons
- [ ] T029 [US1] Create LessonContent API endpoint in `backend/src/api/content.py`:
  - `GET /api/content/lessons/{lesson_id}` returns lesson markdown, metadata, code examples, diagrams
  - `GET /api/content/modules` returns all modules with lesson counts
  - `GET /api/content/modules/{module_id}` returns module details and lessons
- [ ] T030 [US1] Implement frontend lesson page loader in `frontend/src/services/content.service.ts` that:
  - Fetches lesson content from backend API
  - Handles loading/error states
  - Caches content locally for performance
- [ ] T031 [US1] Optimize frontend performance:
  - [ ] Image lazy loading for diagrams
  - [ ] Code splitting for modules (load module only when visited)
  - [ ] Production build optimization (minification, gzip)
- [ ] T032 [US1] Create backend data seeding in `backend/src/db/seed.py` to:
  - Populate 4 modules with metadata
  - Populate 120+ lesson records with content, objectives, code examples (sampled content for MVP)
  - Create lesson cross-references

**Checkpoint**: User Story 1 functional - students can navigate and view all lessons independently

---

## Phase 4: User Story 2 - Query Textbook Content via RAG Chatbot (Priority: P1)

**Goal**: Deploy RAG chatbot (Qdrant + LLM) that retrieves lesson context and generates answers with citations

**Independent Test**: User queries lesson-related questions, receives response within 5 seconds with valid citations, citations link to correct lessons

### Implementation for User Story 2

- [ ] T033 [P] [US2] Create RAG service in `backend/src/services/rag_service.py` with:
  - Query embedding generation (using OpenAI or HuggingFace)
  - Qdrant semantic search (retrieve top 3-5 passages with similarity >0.75)
  - LLM response generation with retrieved context
  - Citation extraction and formatting (format: "[Module, Lesson #.#]")
- [ ] T034 [US2] Implement lesson embedding pipeline in `backend/src/services/lesson_embedding_service.py`:
  - Break each lesson into passages (e.g., per paragraph or section)
  - Generate embeddings for each passage
  - Store in Qdrant with metadata (lesson_id, module_name, passage_text)
  - Provide batch ingest for 120+ lessons
- [ ] T035 [US2] Create chatbot API endpoint in `backend/src/api/chatbot.py`:
  - `POST /api/chatbot/query` accepts user query, optional context (selected text)
  - Returns response with citations, generation time, retrieved passages (for debugging)
  - Logs all queries to ChatbotQuery model for analytics
- [ ] T036 [US2] Implement ChatbotService in `backend/src/services/chatbot_service.py` with:
  - Query validation (check for empty, too-long queries)
  - Out-of-scope detection (if similarity <0.7 for all passages, return "I couldn't find relevant content" message)
  - Response generation timeout handling (max 5 seconds)
  - Citation formatting and link generation
- [ ] T037 [US2] Create ChatbotUI React component in `frontend/src/components/Chatbot/` with:
  - Chat message display (user query + assistant response)
  - Message input form with submit button
  - Text selection context capture (when user selects text and opens chatbot, pre-fill context)
  - Citation link rendering (clickable links to lessons)
  - Loading spinner during response generation
- [ ] T038 [US2] Integrate chatbot into LessonViewer component:
  - Add "Ask" button on lesson pages
  - Allow text selection to open chatbot with context
  - Display chatbot sidebar or modal
- [ ] T039 [US2] Create chatbot service layer in `frontend/src/services/chatbot.service.ts` that:
  - Sends queries to backend API
  - Handles response streaming or polling
  - Manages message history (optional: store recent queries)
- [ ] T040 [US2] Seed Qdrant with lesson embeddings in `backend/src/db/seed_embeddings.py`:
  - Parse all 120+ lesson markdown files
  - Break into passages
  - Generate embeddings
  - Store in Qdrant collection
- [ ] T041 [US2] Create chatbot analytics logging in `backend/src/services/chatbot_service.py`:
  - Log all queries and responses to ChatbotQuery model
  - Track response generation time
  - Track citation accuracy (for later analysis)

**Checkpoint**: User Story 2 functional - chatbot answers questions with citations

---

## Phase 5: User Story 3 - Authenticate and Build a User Profile (Priority: P1)

**Goal**: Enable user signup/login via Better-Auth and store profiles in Neon with specialization fields

**Independent Test**: New user can sign up, complete profile, logout, login, and profile persists with all fields intact

### Implementation for User Story 3

- [ ] T042 [US3] Implement signup endpoint in `backend/src/api/auth.py`:
  - `POST /api/auth/signup` accepts email, optional password, specialization questionnaire
  - Validates email format, password strength (if provided)
  - Creates User record via Better-Auth
  - Stores profile data (ros_experience_level, focus_area, language_preference) in User model
  - Returns session token on success
  - Rate limits to 5 accounts per IP per hour
- [ ] T043 [US3] Implement login endpoint in `backend/src/api/auth.py`:
  - `POST /api/auth/login` accepts email, password
  - Validates credentials via Better-Auth
  - Returns session token on success
  - Returns 401 on invalid credentials
- [ ] T044 [US3] Implement profile endpoints in `backend/src/api/auth.py`:
  - `GET /api/auth/profile` returns authenticated user's profile (protected)
  - `PATCH /api/auth/profile` updates user specialization, focus_area, language_preference (protected)
  - `POST /api/auth/logout` invalidates session token
- [ ] T045 [US3] Create UserAuthService in `backend/src/services/auth_service.py` with:
  - Integration with Better-Auth for credential management
  - Session token generation (30-day expiry, configurable)
  - Password hashing (handled by Better-Auth)
  - Email validation
  - Rate limiting on signup IP
- [ ] T046 [US3] Create signup form component in `frontend/src/components/Auth/SignupForm.tsx` with:
  - Email input with validation
  - Password input (optional) with strength indicator
  - Specialization questionnaire (checkboxes/radio for ROS level, hardware/software focus)
  - Submit button and loading state
  - Success/error messages
- [ ] T047 [US3] Create login form component in `frontend/src/components/Auth/LoginForm.tsx` with:
  - Email input
  - Password input
  - Submit button and loading state
  - Error message display (incorrect credentials)
  - Link to signup
- [ ] T048 [US3] Create authentication context in `frontend/src/context/AuthContext.tsx` with:
  - Current user state
  - Session token management (store in localStorage or secure cookie)
  - Login/logout functions
  - Auto-login on page load (if valid token exists)
- [ ] T049 [US3] Create profile page component in `frontend/src/components/Auth/ProfilePage.tsx` with:
  - Display user email
  - Editable profile fields (specialization, ros_experience_level, focus_area, language_preference)
  - Update/save button
  - Logout button
  - Success/error messages on update
- [ ] T050 [US3] Implement authentication middleware in frontend routing:
  - Redirect unauthenticated users to login page on protected routes
  - Redirect authenticated users away from signup/login pages
  - Show user menu (email, profile link, logout) in header when authenticated
- [ ] T051 [US3] Create auth service in `frontend/src/services/auth.service.ts` that:
  - Calls backend signup/login/profile endpoints
  - Manages session token in localStorage (or secure storage)
  - Provides user state to components
  - Handles session expiry/refresh

**Checkpoint**: User Story 3 functional - users can sign up, login, and manage profiles

---

## Phase 6: User Story 4 - Receive Personalized Content Recommendations (Priority: P2)

**Goal**: Generate learning path recommendations based on user profile (specialization, experience level)

**Independent Test**: Two users with different profiles receive distinct recommendations; recommendations align with profiles

### Implementation for User Story 4

- [ ] T052 [US4] Create PersonalizationService in `backend/src/services/personalization_service.py` with:
  - Recommendation algorithm based on user ros_experience_level and specialization tags
  - Query user's completed lessons
  - Filter lessons by difficulty matching experience level
  - Rank lessons by alignment with specialization and prerequisite completion
  - Return top 10 recommendations
- [ ] T053 [US4] Create recommendation caching in `backend/src/services/personalization_service.py`:
  - Cache recommendations per user for 24 hours
  - Invalidate cache when user completes a lesson or updates profile
  - Reduce computation overhead
- [ ] T054 [US4] Implement recommendations endpoint in `backend/src/api/personalization.py`:
  - `GET /api/personalization/recommendations` (protected) returns list of recommended lessons with explanations
  - Include explanation text ("Recommended because it builds on your ROS fundamentals")
  - Order by recommendation score
- [ ] T055 [US4] Create engagement tracking endpoint in `backend/src/api/personalization.py`:
  - `POST /api/personalization/engagement` (protected) logs lesson view/completion events
  - Accepts lesson_id, event_type (view, completion, time_spent)
  - Stores in UserProgress model
  - Triggers recommendation cache invalidation
- [ ] T056 [US4] Implement RecommendationWidget React component in `frontend/src/components/Personalization/RecommendationWidget.tsx` with:
  - Display "Recommended for You" section on module roadmap
  - Show top 5 recommendations with explanation text
  - Visual distinction (highlight/badge) on recommended lessons
  - Click to navigate to lesson
- [ ] T057 [US4] Integrate recommendations into ModuleRoadmap component:
  - Fetch recommendations API on user login
  - Display "Recommended" badge on matching lessons in roadmap
  - Show hover tooltip with recommendation explanation
- [ ] T058 [US4] Create engagement tracking in LessonViewer component:
  - Log lesson view event when lesson loads
  - Log lesson completion when user clicks "Next" button
  - Track time spent on lesson
  - Send events to backend engagement endpoint

**Checkpoint**: User Story 4 functional - users receive tailored recommendations

---

## Phase 7: User Story 5 - Access Content in Multiple Languages (Priority: P3)

**Goal**: Support English, Spanish, Mandarin with identical code examples and fallback to English

**Independent Test**: Select language, verify all content translated, code examples identical, language preference persists

### Implementation for User Story 5

- [ ] T059 [US5] Create TranslationService in `backend/src/services/translation_service.py` with:
  - Batch translation of lesson content to Spanish and Mandarin
  - Code example preservation (regex to skip code blocks)
  - Diagram reference preservation
  - Translation quality checks (100% match for code)
- [ ] T060 [US5] Create translation storage in database:
  - Populate ContentTranslation model with ES (Spanish) and ZH (Mandarin) translations
  - Include reviewed_at field for translation quality tracking
  - Create migration to seed initial translations
- [ ] T061 [US5] Implement language preference in user profile:
  - Add language_preference field to User model (default: "en")
  - Update profile PATCH endpoint to accept language_preference changes
  - Store in Neon
- [ ] T062 [US5] Create translations endpoint in `backend/src/api/content.py`:
  - `GET /api/content/lessons/{lesson_id}?lang=es` returns lesson in Spanish (or English fallback)
  - `GET /api/content/languages` returns list of supported languages
  - Fallback to English if translation unavailable (with banner notification)
- [ ] T063 [US5] Create language selector component in `frontend/src/components/i18n/LanguageSelector.tsx` with:
  - Dropdown with English, Spanish, Mandarin options
  - Display current selected language
  - Sync with user profile language_preference
  - Trigger page reload on language change
- [ ] T064 [US5] Implement RTL support for Arabic/Hebrew (if supported):
  - Add `dir="rtl"` CSS to language selector
  - Test layout in RTL mode
- [ ] T065 [US5] Add language preference persistence:
  - Save language_preference to user profile on change
  - Load from profile on login
  - Store in localStorage as fallback for guests
- [ ] T066 [US5] Create translation quality validation in `backend/src/db/seed_translations.py`:
  - Verify 100% code example match between English and translated versions
  - Verify diagram references identical
  - Log any mismatches for manual review

**Checkpoint**: User Story 5 functional - multi-language support operational

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements affecting multiple user stories, optimizations, and final quality validation

- [ ] T067 [P] Documentation updates:
  - [ ] T067a [P] Create `docs/ARCHITECTURE.md` with system diagram and component explanation
  - [ ] T067b [P] Create `docs/DEV_SETUP.md` with local environment setup guide (database, services, secrets)
  - [ ] T067c [P] Create `docs/DEPLOYMENT.md` with production deployment guide
  - [ ] T067d [P] Create `docs/API_CONTRACTS.md` with OpenAPI reference for all endpoints
  - [ ] T067e [P] Create `docs/CODE_EXAMPLES.md` documenting how to add/verify code examples
- [ ] T068 Code cleanup and standards enforcement:
  - [ ] Backend: Run Black formatter, Ruff linter, fix all violations
  - [ ] Frontend: Run ESLint and Prettier, fix all violations
- [ ] T069 Performance optimization across all stories:
  - [ ] Frontend: Measure and optimize Largest Contentful Paint (LCP), First Input Delay (FID), Cumulative Layout Shift (CLS)
  - [ ] Backend: Optimize database queries (N+1 query detection), add caching where appropriate
  - [ ] CDN: Configure static asset caching headers
- [ ] T070 Security hardening:
  - [ ] Add HTTPS-only enforcement (HSTS headers)
  - [ ] Add CSRF protection to forms
  - [ ] Add input sanitization for user-provided content (chatbot queries, profile updates)
  - [ ] Add SQL injection prevention (SQLAlchemy ORM provides this, validate)
  - [ ] Audit Better-Auth configuration for security best practices
- [ ] T071 Monitoring and observability setup:
  - [ ] Add structured logging (JSON format) to backend services
  - [ ] Create dashboard for key metrics (uptime, response times, error rates)
  - [ ] Setup alerts for critical errors and performance degradation
- [ ] T072 Accessibility compliance:
  - [ ] Run Axe accessibility audit on all pages
  - [ ] Fix WCAG 2.1 AA violations (color contrast, alt text, keyboard navigation)
  - [ ] Test with screen readers (NVDA, JAWS)
- [ ] T073 [P] Additional unit tests (beyond core TDD):
  - [ ] `backend/tests/unit/test_*_service.py` for all service classes
  - [ ] `backend/tests/unit/test_*_schema.py` for Pydantic schema validation
  - [ ] `frontend/tests/unit/test_*.spec.ts` for component logic
- [ ] T074 Run quickstart.md validation:
  - [ ] Document must have step-by-step setup instructions
  - [ ] Test that a new developer can follow instructions and get the system running locally
  - [ ] Update with any missing steps or clarifications

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational (Phase 2) completion
  - User stories can then proceed in priority order (P1 → P2 → P3)
  - Or proceed in parallel if team capacity allows
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: No dependencies on other stories - can start after Phase 2
- **User Story 2 (P1)**: Depends on US1 (needs lessons to be indexed) - start after US1 content exists
- **User Story 3 (P1)**: No dependencies on other stories - can start in parallel with US1-US2
- **User Story 4 (P2)**: Depends on US3 (needs user profiles) - start after US3 complete
- **User Story 5 (P3)**: Depends on US1 (needs lesson content) - lowest priority, start after P1 stories done

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1 (Setup)**: All tasks marked [P] can run in parallel
- **Phase 2 (Foundational)**:
  - Database setup (T008-T010) can run in parallel with backend scaffolding (T011-T021)
  - All ORM model definitions (T009a-T009g) can run in parallel
  - All schema definitions (T013a-T013e) can run in parallel
  - All OpenAPI contract files (T021a-T021e) can run in parallel
- **Phase 3 (US1)**:
  - All module pages (T022a-T022d) can run in parallel
  - All lesson markdown files (T023-T026) can run in parallel with component development (T027-T032)
- **Phase 3-7 (US1-US5)**:
  - Different user stories can be worked on in parallel once Phase 2 is complete
  - US1 and US3 can run in parallel (no dependencies)
  - US1 and US2 sequential within the team but could use multiple developers
  - US4 and US5 can start immediately after Phase 2 once US1 and US3 have basic structure
- **Phase 8 (Polish)**: All documentation tasks marked [P] can run in parallel

---

## Implementation Strategy

### MVP First (User Stories 1-3 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Textbook content)
4. Complete Phase 4: User Story 2 (RAG chatbot)
5. Complete Phase 5: User Story 3 (Authentication)
6. **STOP and VALIDATE**: Test User Stories 1-3 independently, deploy if ready
7. Gather feedback from users before proceeding to US4-US5

### Incremental Delivery (MVP + US4 + US5)

1. Complete phases 1-5 (MVP: US1-US3)
2. Test and validate MVP
3. Deploy MVP to production
4. Collect user feedback
5. Complete Phase 6: User Story 4 (Personalization)
6. Test and validate US4
7. Deploy US4 to production
8. Complete Phase 7: User Story 5 (Translation)
9. Test and validate US5
10. Deploy US5 to production
11. Complete Phase 8: Polish & optimization

### Parallel Team Strategy

With multiple developers:

1. **Developer A**: Phase 1 (setup) + Phase 2a (database setup) + Phase 3 (US1 - textbook content)
2. **Developer B**: Phase 2b (backend scaffolding + auth) + Phase 4 (US2 - RAG chatbot)
3. **Developer C**: Phase 5 (US3 - authentication forms + profile) + Phase 6 (US4 - personalization)
4. **Developer D** (if available): Phase 7 (US5 - translation) + Phase 8 (Polish)

Once Phase 2 is complete, developers can work on parallel user stories independently.

---

## Notes

- [P] tasks = different files, no dependencies between them
- [Story] label (US1, US2, etc.) maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (if using TDD approach)
- Commit after each task or logical group (e.g., after completing all models T009a-T009g)
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

---

## Success Criteria Alignment

Each task above directly supports one or more of the 30 Success Criteria (SC-001 through SC-030):

| Task Group | Maps to Success Criteria |
|-----------|---------------------------|
| T022-T032 (US1 content) | SC-001, SC-002, SC-003, SC-004, SC-005, SC-021 |
| T033-T041 (US2 chatbot) | SC-006, SC-007, SC-008, SC-009, SC-010 |
| T042-T051 (US3 auth) | SC-011, SC-012, SC-013, SC-014 |
| T052-T058 (US4 personalization) | SC-015, SC-016, SC-017 |
| T059-T066 (US5 translation) | SC-018, SC-019, SC-020 |
| T067-T074 (Polish) | SC-021, SC-022, SC-023, SC-024, SC-029, SC-030 |
