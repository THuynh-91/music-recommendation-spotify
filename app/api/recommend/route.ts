import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import {
  applySpotifyCookies,
  ensureAccessToken,
  refreshSpotifyTokens,
} from "@/lib/server-auth";
import type { SpotifyTokenResponse } from "@/lib/server-auth";

interface RecommendRequestBody {
  url?: string;
  limit?: number;
}

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

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
    const requestBody: RecommendRequestBody = { url: payload.url, limit };

    const serviceToken = process.env.RECOMMENDER_SERVICE_TOKEN;
    if (!serviceToken) {
      return NextResponse.json({ error: "misconfigured_service_token" }, { status: 500 });
    }
    const backendBase = process.env.RECOMMENDER_API_URL ?? DEFAULT_BACKEND_URL;

    const { accessToken: initialToken, tokens: refreshedTokens } = await ensureAccessToken();
    if (!initialToken) {
      return NextResponse.json({ error: "not_authenticated" }, { status: 401 });
    }

    const cookieStore = cookies();
    let accessToken = initialToken;
    let tokensToPersist: SpotifyTokenResponse | undefined = refreshedTokens;
    let backendResponse = await callBackend(backendBase, serviceToken, accessToken, requestBody);

    if (backendResponse.status === 401) {
      const refreshToken = tokensToPersist?.refresh_token || cookieStore.get("sp_refresh_token")?.value || null;
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