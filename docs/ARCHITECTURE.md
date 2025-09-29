# Architecture Overview

## High-Level System
- **Next.js 14 App Router** drives the UI and Spotify OAuth flow. It keeps HTTP-only cookies for access/refresh tokens and proxies secure calls to the backend.
- **FastAPI service** performs data ingestion, feature synthesis, FAISS vector search, and business logic for explanations.
- **PostgreSQL** persists normalized Spotify entities (artists, albums, tracks, playlists) and computed feature vectors so that indexes can be rebuilt.
- **Redis** caches volatile lookups (Spotify metadata, token exchanges, FAISS warm cache) and coordinates ingestion locks.
- **FAISS** (`IndexIVFFlat` w/ cosine) powers the ANN search surface for recommendations.

## Data Flow
1. The user authenticates with Spotify via PKCE. Tokens are stored in HTTP-only cookies by `app/api/auth/*` routes.
2. The UI calls `POST /api/recommend` with a Spotify track or playlist URL.
3. That route ensures a valid access token (refreshing when needed) and forwards the request to the FastAPI backend with a short-lived service JWT.
4. The FastAPI backend:
   - Parses the URL, determines entity type, and pulls Spotify metadata + audio features/analysis through the user's access token.
   - For playlists it normalizes all tracks, stores them in Postgres, updates the FAISS index, and returns a catalog summary.
   - For tracks it computes a feature vector (merging Spotify features, structural metrics from the audio analysis, and optional DSP descriptors derived with librosa if preview audio is available).
   - The vector is normalized and searched against FAISS. Top-k hits are enriched with metadata, similarity scores, and lightweight explanations.
5. The Next.js route returns the recommendations to the UI.

## Backend Composition
- **`backend/app/main.py`**: FastAPI app factory, middleware, startup/shutdown hooks.
- **`backend/app/api/routes`**: modular routers (`auth`, `ingest`, `recommend`).
- **`backend/app/spotify`**: async client for Spotify Web API with automatic retry and rate limiting backoff.
- **`backend/app/services/features.py`**: vector synthesis and DSP computation.
- **`backend/app/services/index.py`**: FAISS index lifecycle (load/save/rebuild).
- **`backend/app/db`**: SQLAlchemy models + async session management.
- **`backend/app/cache`**: Redis utilities and token cache helpers.

## Operations
- `docker-compose.next.yml` orchestrates Next.js, FastAPI, Postgres, and Redis. Both web containers run on non-root users and use multi-stage Dockerfiles.
- Health endpoints exist at `/api/health` (Next) and `/api/v1/health` (FastAPI).
- Background task rebuilds FAISS nightly from DB persistent vectors.
- Structured logging (JSON) is configured for production with OpenTelemetry-ready hooks.

## Security Considerations
- Secrets flow through `.env` and Docker secrets (when present). Spotify client secret is only required by the backend.
- All inter-service calls are authenticated with HMAC-signed service tokens to prevent spoofing.
- Input URLs are sanitized and validated against a strict Spotify pattern before processing.
- Long-running ingestion jobs are guarded via Redis distributed locks to avoid duplicate work.

## Testing Strategy
- Backend contains pytest suites with `httpx.AsyncClient` exercising the main flows and fixtures for Spotify API stubs.
- Frontend relies on Playwright component tests for core flows and React Testing Library for UI state.
- CI (GitHub Actions template) runs lint, type-check, unit tests, and builds Docker images.