import { NextResponse } from "next/server";

import {
  applySpotifyCookies,
  ensureAccessToken,
  refreshSpotifyTokens,
} from "@/lib/server-auth";
import type { SpotifyTokenResponse } from "@/lib/server-auth";

const PROFILE_ENDPOINT = "https://api.spotify.com/v1/me";

async function fetchProfile(token: string) {
  return fetch(PROFILE_ENDPOINT, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
    cache: "no-store",
  });
}

export async function GET() {
  try {
    const { accessToken, tokens } = await ensureAccessToken();
    if (!accessToken) {
      return NextResponse.json({ error: "not_authenticated" }, { status: 401 });
    }

    let tokensToPersist: SpotifyTokenResponse | undefined = tokens;
    let token = accessToken;
    let profileResponse = await fetchProfile(token);

    if (profileResponse.status === 401) {
      const refreshToken = tokensToPersist?.refresh_token;
      if (refreshToken) {
        const refreshed = await refreshSpotifyTokens(refreshToken);
        tokensToPersist = refreshed;
        token = refreshed.access_token;
        profileResponse = await fetchProfile(token);
      }
    }

    const data = await profileResponse.json().catch(() => ({}));
    const response = NextResponse.json(data, { status: profileResponse.status });
    if (tokensToPersist) {
      applySpotifyCookies(response, tokensToPersist);
    }
    return response;
  } catch (error) {
    const message = error instanceof Error ? error.message : "server_error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}