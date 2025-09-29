# Spotify Recommendation Platform

A full-stack Spotify intelligence platform that authenticates with Spotify, ingests tracks and playlists, and surfaces explainable recommendations using FAISS vector search and DSP feature engineering.

## Features
- **PKCE OAuth** with Spotify and secure token storage via HTTP-only cookies.
- Track & playlist ingestion via FastAPI, SQLAlchemy, Redis caching, and FAISS-based similarity search.
- DSP descriptors (librosa) blended with Spotify audio features and analysis for richer vectors.
- Genre-aware re-ranking and diversity guardrails to avoid repetitive recommendations.
- Next.js 14 App Router UI with modern styling and production-quality DX (lint, type-check, Docker).
- Containerised stack (Next.js frontend, FastAPI backend, Postgres, Redis) orchestrated by Docker Compose.

## Project Layout
```
.
+-- app/                     # Next.js App Router code
+-- backend/                 # FastAPI service
+-- docs/                    # Architecture notes
+-- lib/                     # Shared frontend utilities
+-- types/                   # Type declarations
+-- docker-compose.next.yml  # Production docker composition
```

## Prerequisites
- Node.js 20+
- Python 3.11+
- Docker & Docker Compose (for containerised runs)
- Spotify application with redirect URI `http://127.0.0.1:3000/callback`

Populate `.env` with your secrets (see [.env](./.env) for required keys):
```
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
REDIRECT_URI=http://127.0.0.1:3000/callback
RECOMMENDER_SERVICE_TOKEN=change-me
BACKEND_SERVICE_TOKEN=change-me # should match RECOMMENDER_SERVICE_TOKEN
```

## Local Development
### Frontend
```bash
npm install
npm run dev
```
Runs on [http://127.0.0.1:3000](http://127.0.0.1:3000).

### Backend
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate # or source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Ensure Postgres (`postgresql+asyncpg://postgres:postgres@localhost:5432/spotify`) and Redis are available in development. You can spin them up quickly with:
```bash
docker run --rm -p 5432:5432 -e POSTGRES_DB=spotify -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres postgres:16-alpine
docker run --rm -p 6379:6379 redis:7-alpine
```

### Tooling
- `npm run lint` / `npm run build` for Next.js validation.
- `python -m compileall backend/app` to sanity-check backend syntax.
- (Optional) add `pytest` suites under `backend/tests` for deeper coverage.

## Production via Docker Compose
```bash
docker compose -f docker-compose.next.yml up --build
```
Services exposed:
- Frontend: `http://127.0.0.1:3000`
- Backend API: `http://127.0.0.1:8000/v1/...`
- Postgres (`localhost:5432`) and Redis (`localhost:6379`) for persistence & caching.

Images are multi-stage, non-root, and produce a standalone Next.js server bundle plus a slim Python runtime with FAISS, librosa, and friends baked in.

## API Highlights
- `POST /v1/recommendations` – ingest playlist or fetch track recommendations; returns explanations & similarity scores.
- `GET /v1/health` – backend health & FAISS index status.
- Frontend proxy endpoint `/api/recommend` handles token refresh and forwards to backend with service-token auth.

## Security Considerations
- Spotify tokens kept exclusively in HTTP-only cookies; server routes refresh on-demand.
- Frontend?backend authenticated via shared service token (`RECOMMENDER_SERVICE_TOKEN`).
- Redis locks de-duplicate playlist ingestion and FAISS index rebuilds.

## Monitoring & Next Steps
- Add pytest suites for backend recommendation pipelines.
- Wire Playwright or Vitest UI tests for critical flows.
- Extend CI to run `npm run build`, backend compile, and Docker image builds.
- Deploy containers (e.g., Render, Fly.io, AWS ECS) with secret management for Spotify credentials.