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

export function applySpotifyCookies(res: NextResponse, tokens: SpotifyTokenResponse) {
  const isProd = process.env.NODE_ENV === "production";
  if (tokens.access_token) {
    res.cookies.set("sp_access_token", tokens.access_token, {
      httpOnly: true,
      secure: isProd,
      sameSite: "lax",
      path: "/",
      maxAge: tokens.expires_in ?? 3600,
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
  return json as SpotifyTokenResponse;
}

export async function ensureAccessToken(): Promise<{
  accessToken: string | null;
  tokens?: SpotifyTokenResponse;
}> {
  const cookieStore = cookies();
  const access = cookieStore.get("sp_access_token")?.value;
  if (access) {
    return { accessToken: access };
  }
  const refresh = cookieStore.get("sp_refresh_token")?.value;
  if (!refresh) {
    return { accessToken: null };
  }
  const tokens = await refreshSpotifyTokens(refresh);
  return { accessToken: tokens.access_token ?? null, tokens };
}