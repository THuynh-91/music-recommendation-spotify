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
- Docker & Docker Compose (only for the full containerised live stack)
- A Spotify developer application (see below) for the **live** flow

### What YOU must provide for the live flow
This repo cannot create a Spotify app for you. To run the real (non-demo) flow you must:

1. Create an app at <https://developer.spotify.com/dashboard>.
2. Copy its **Client ID** into `SPOTIFY_CLIENT_ID` and `NEXT_PUBLIC_SPOTIFY_CLIENT_ID`
   (and `SPOTIFY_CLIENT_SECRET` for the backend client-credentials fallback).
3. In the dashboard, register a **Redirect URI** that EXACTLY matches your
   frontend URL + `/callback`. If you run the dev server on port 4200, register
   `http://127.0.0.1:4200/callback` and set `REDIRECT_URI` /
   `NEXT_PUBLIC_REDIRECT_URI` to the same value. (A mismatch is the #1 cause of
   `INVALID_CLIENT: Invalid redirect URI`.)
4. Pick any shared secret and set BOTH `RECOMMENDER_SERVICE_TOKEN` (frontend) and
   `BACKEND_SERVICE_TOKEN` (backend) to the same value — they authenticate the
   frontend→backend hop.
5. Point `RECOMMENDER_API_URL` at the running backend (e.g. `http://127.0.0.1:4201`).

```
SPOTIFY_CLIENT_ID=...
NEXT_PUBLIC_SPOTIFY_CLIENT_ID=...        # same value
SPOTIFY_CLIENT_SECRET=...
REDIRECT_URI=http://127.0.0.1:4200/callback
NEXT_PUBLIC_REDIRECT_URI=http://127.0.0.1:4200/callback   # same value, MUST match dashboard
RECOMMENDER_API_URL=http://127.0.0.1:4201
RECOMMENDER_SERVICE_TOKEN=some-shared-secret
BACKEND_SERVICE_TOKEN=some-shared-secret  # must equal RECOMMENDER_SERVICE_TOKEN
```

> **Spotify API note:** As of Nov 2024 Spotify deprecated several endpoints this
> backend uses (`/recommendations`, `/audio-features`, `/audio-analysis`,
> related-artists) for *newly created* apps. If your app only has access to the
> current Web API, the live recommender will return few/no results. The code is
> wired correctly; this is a Spotify platform limitation outside this repo.

### Demo mode (no Spotify app required)
Set `DEMO_MODE=1` for the frontend and `BACKEND_DEMO_MODE=true` for the backend
to run the entire UI + API with deterministic mock data — no Spotify app,
Postgres, Redis, FAISS, or librosa needed. The UI shows a "Demo mode" banner and
still exercises login-state, profile, and recommendation rendering.

## Local Development
### Frontend
```bash
npm install
npm run dev -- -p 4200          # live
DEMO_MODE=1 npm run dev -- -p 4200   # demo (no Spotify needed)
```
Runs on [http://127.0.0.1:4200](http://127.0.0.1:4200). Keep this port consistent
with your registered Spotify redirect URI.

### Backend
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate # or source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 4201             # live (needs Postgres + Redis)
BACKEND_DEMO_MODE=true uvicorn app.main:app --port 4201   # demo (no infra)
```
In demo mode the backend boots without Postgres, Redis, FAISS, or librosa.
`faiss` and `librosa` are imported lazily, so the service starts even if those
heavy wheels aren't installed (they're only needed for live DSP/vector search).

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
- `POST /v1/recommendations` � ingest playlist or fetch track recommendations; returns explanations & similarity scores.
- `GET /v1/health` � backend health & FAISS index status.
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