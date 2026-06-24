import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export interface SpotifyTokenResponse {
  access_token: string;
  token_type?: string;
  scope?: string;
  expires_in?: number;
  refresh_token?: string;
}

const SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token";

// Refresh proactively when the access token has this many seconds (or fewer) left.
const NEAR_EXPIRY_THRESHOLD_SECONDS = 120;

export function applySpotifyCookies(res: NextResponse, tokens: SpotifyTokenResponse) {
  const isProd = process.env.NODE_ENV === "production";
  const expiresIn = tokens.expires_in ?? 3600;
  if (tokens.access_token) {
    res.cookies.set("sp_access_token", tokens.access_token, {
      httpOnly: true,
      secure: isProd,
      sameSite: "lax",
      path: "/",
      maxAge: expiresIn,
    });
    // Persist the absolute expiry so we can refresh *before* the token dies.
    const expiresAt = Math.floor(Date.now() / 1000) + expiresIn;
    res.cookies.set("sp_access_expires_at", String(expiresAt), {
      httpOnly: true,
      secure: isProd,
      sameSite: "lax",
      path: "/",
      maxAge: expiresIn,
    });
  }
  if (tokens.refresh_token) {
    res.cookies.set("sp_refresh_token", tokens.refresh_token, {
      httpOnly: true,
      secure: isProd,
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24 * 30,
    });
  }
}

export async function refreshSpotifyTokens(refreshToken: string): Promise<SpotifyTokenResponse> {
  const clientId = process.env.SPOTIFY_CLIENT_ID;
  if (!clientId) {
    throw new Error("SPOTIFY_CLIENT_ID not configured");
  }
  const params = new URLSearchParams({
    grant_type: "refresh_token",
    refresh_token: refreshToken,
    client_id: clientId,
  });
  const response = await fetch(SPOTIFY_TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: params.toString(),
    cache: "no-store",
  });
  const json = await response.json();
  if (!response.ok) {
    throw new Error(json?.error_description || "failed_to_refresh_token");
  }
  // Spotify often omits refresh_token on refresh; reuse the previous one so it
  // gets re-persisted and the long-lived session survives.
  const tokens = json as SpotifyTokenResponse;
  if (!tokens.refresh_token) {
    tokens.refresh_token = refreshToken;
  }
  return tokens;
}

/**
 * Resolve a usable access token. Refreshes proactively when the current token is
 * absent OR within NEAR_EXPIRY_THRESHOLD_SECONDS of expiring, so long-running
 * ingestions don't die mid-flight on a 401.
 */
export async function ensureAccessToken(): Promise<{
  accessToken: string | null;
  tokens?: SpotifyTokenResponse;
}> {
  const cookieStore = cookies();
  const access = cookieStore.get("sp_access_token")?.value;
  const refresh = cookieStore.get("sp_refresh_token")?.value;
  const expiresAtRaw = cookieStore.get("sp_access_expires_at")?.value;

  const now = Math.floor(Date.now() / 1000);
  const expiresAt = expiresAtRaw ? Number.parseInt(expiresAtRaw, 10) : NaN;
  const isNearExpiry =
    Number.isFinite(expiresAt) && expiresAt - now <= NEAR_EXPIRY_THRESHOLD_SECONDS;

  // Token is present and comfortably valid.
  if (access && !isNearExpiry) {
    return { accessToken: access };
  }

  // Token absent or near expiry -> refresh if we can.
  if (refresh) {
    try {
      const tokens = await refreshSpotifyTokens(refresh);
      return { accessToken: tokens.access_token ?? null, tokens };
    } catch (err) {
      // If we still have a (near-expiry but non-empty) token, fall back to it
      // rather than hard-failing the request.
      if (access) {
        return { accessToken: access };
      }
      throw err;
    }
  }

  // No refresh token; use whatever access token we have (may be null).
  return { accessToken: access ?? null };
}
