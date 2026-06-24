import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { applySpotifyCookies } from "@/lib/server-auth";
import type { SpotifyTokenResponse } from "@/lib/server-auth";

function b64urlDecode(s: string) {
  s = (s || "").replace(/-/g, "+").replace(/_/g, "/");
  const pad = (4 - (s.length % 4)) % 4;
  s += "=".repeat(pad);
  return Buffer.from(s, "base64").toString("utf8");
}

export async function POST(req: Request) {
  try {
    const { code, state } = await req.json();
    if (!code) return NextResponse.json({ error: "missing_code" }, { status: 400 });
    if (!state) return NextResponse.json({ error: "missing_state" }, { status: 400 });

    // CSRF / state validation: the `state` returned by Spotify must match the
    // one we minted in /api/auth/start (stored in an HTTP-only cookie). This
    // prevents an attacker from injecting their own code/state pair.
    const cookieStore = cookies();
    const expectedState = cookieStore.get("pkce_state")?.value;
    if (!expectedState || expectedState !== state) {
      return NextResponse.json({ error: "state_mismatch" }, { status: 400 });
    }

    // Prefer the verifier from the trusted HTTP-only cookie set in
    // /api/auth/start; fall back to the copy embedded in `state` for resilience.
    let verifier = cookieStore.get("pkce_verifier")?.value ?? "";
    if (!verifier) {
      try {
        const decoded = JSON.parse(b64urlDecode(state));
        verifier = typeof decoded?.v === "string" ? decoded.v : "";
      } catch {}
    }

    if (!verifier) return NextResponse.json({ error: "missing_verifier_from_state" }, { status: 400 });

    const params = new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: process.env.REDIRECT_URI ?? "",
      client_id: process.env.SPOTIFY_CLIENT_ID ?? "",
      code_verifier: verifier,
    });

    const r = await fetch("https://accounts.spotify.com/api/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: params.toString(),
    });

    const data = (await r.json()) as SpotifyTokenResponse & { expires_in: number };
    if (!r.ok) return NextResponse.json(data, { status: r.status });

    const res = NextResponse.json({ ok: true, expires_in: data.expires_in });
    applySpotifyCookies(res, data);
    // One-time PKCE values are now spent; clear them so they can't be replayed.
    const isProd = process.env.NODE_ENV === "production";
    const clear = { httpOnly: true, secure: isProd, sameSite: "lax" as const, path: "/", maxAge: 0 };
    res.cookies.set({ name: "pkce_verifier", value: "", ...clear });
    res.cookies.set({ name: "pkce_state", value: "", ...clear });
    return res;
  } catch (error) {
    const message = error instanceof Error ? error.message : "server_error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
