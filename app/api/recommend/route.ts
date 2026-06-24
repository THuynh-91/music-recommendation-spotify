import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import {
  applySpotifyCookies,
  ensureAccessToken,
  refreshSpotifyTokens,
} from "@/lib/server-auth";
import type { SpotifyTokenResponse } from "@/lib/server-auth";
import { buildDemoRecommendations } from "@/lib/demo";

interface RecommendRequestBody {
  url?: string;
  limit?: number;
  market?: string;
}

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";
const PROFILE_ENDPOINT = "https://api.spotify.com/v1/me";

function isDemoMode() {
  return (
    process.env.DEMO_MODE === "1" ||
    !process.env.SPOTIFY_CLIENT_ID ||
    !process.env.RECOMMENDER_SERVICE_TOKEN
  );
}

/** Resolve the user's Spotify market (ISO country) for region-correct results. */
async function resolveMarket(accessToken: string): Promise<string | undefined> {
  try {
    const res = await fetch(PROFILE_ENDPOINT, {
      headers: { Authorization: `Bearer ${accessToken}` },
      cache: "no-store",
    });
    if (!res.ok) return undefined;
    const data = (await res.json()) as { country?: string };
    return typeof data.country === "string" && data.country.length === 2
      ? data.country
      : undefined;
  } catch {
    return undefined;
  }
}

async function callBackend(
  baseUrl: string,
  serviceToken: string,
  accessToken: string,
  body: RecommendRequestBody,
) {
  const endpoint = new URL("/v1/recommendations", baseUrl);
  return fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
      "X-Service-Token": serviceToken,
    },
    body: JSON.stringify(body),
    cache: "no-store",
  });
}

export async function POST(req: Request) {
  try {
    const payload = (await req.json().catch(() => ({}))) as RecommendRequestBody;
    if (!payload.url) {
      return NextResponse.json({ error: "missing_url" }, { status: 400 });
    }

    const limit =
      typeof payload.limit === "number"
        ? Math.min(Math.max(Math.trunc(payload.limit), 1), 50)
        : undefined;

    // DEGRADED / DEMO PATH: no real Spotify app configured -> return mock data
    // so the UI is fully exercisable without credentials.
    if (isDemoMode()) {
      return NextResponse.json(buildDemoRecommendations(payload.url, limit ?? 15));
    }

    const serviceToken = process.env.RECOMMENDER_SERVICE_TOKEN as string;
    const backendBase = process.env.RECOMMENDER_API_URL ?? DEFAULT_BACKEND_URL;

    const { accessToken: initialToken, tokens: refreshedTokens } = await ensureAccessToken();
    if (!initialToken) {
      return NextResponse.json({ error: "not_authenticated" }, { status: 401 });
    }

    const cookieStore = cookies();
    let accessToken = initialToken;
    let tokensToPersist: SpotifyTokenResponse | undefined = refreshedTokens;

    // Derive market from the user's country instead of hardcoding "US".
    const market = payload.market ?? (await resolveMarket(accessToken));

    const requestBody: RecommendRequestBody = { url: payload.url };
    if (typeof limit === "number") requestBody.limit = limit;
    if (market) requestBody.market = market;

    let backendResponse = await callBackend(backendBase, serviceToken, accessToken, requestBody);

    if (backendResponse.status === 401) {
      const refreshToken =
        tokensToPersist?.refresh_token || cookieStore.get("sp_refresh_token")?.value || null;
      if (refreshToken) {
        const newTokens = await refreshSpotifyTokens(refreshToken);
        tokensToPersist = newTokens;
        accessToken = newTokens.access_token;
        backendResponse = await callBackend(backendBase, serviceToken, accessToken, requestBody);
      }
    }

    const data = await backendResponse.json().catch(() => ({}));
    const response = NextResponse.json(data, { status: backendResponse.status });
    if (tokensToPersist) {
      applySpotifyCookies(response, tokensToPersist);
    }
    return response;
  } catch (error) {
    const message = error instanceof Error ? error.message : "server_error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
