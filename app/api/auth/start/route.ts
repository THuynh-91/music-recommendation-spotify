import { NextResponse } from "next/server";

function randomString(len = 64) {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
  let out = "";
  for (let i = 0; i < len; i++) out += chars[Math.floor(Math.random() * chars.length)];
  return out;
}
function base64urlEncode(s: string) {
  return Buffer.from(s, "utf8").toString("base64").replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"");
}
function base64url(buf: ArrayBuffer) {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return Buffer.from(s, "binary").toString("base64").replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"");
}
async function sha256(s: string) {
  const enc = new TextEncoder();
  const digest = await crypto.subtle.digest("SHA-256", enc.encode(s));
  return base64url(digest);
}

export async function GET() {
  const clientId = process.env.SPOTIFY_CLIENT_ID ?? "";
  const redirectUri = process.env.REDIRECT_URI ?? "";
  if (!clientId || !redirectUri) {
    return NextResponse.json({ error: "missing_server_env" }, { status: 500 });
  }

  const verifier = randomString(64);
  const challenge = await sha256(verifier);
  const nonce = randomString(24);

  // Embed verifier in state (plus a nonce)
  const statePayload = { v: verifier, n: nonce };
  const state = base64urlEncode(JSON.stringify(statePayload));

  const scope = [
    "user-read-email",
    "user-read-private",
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-library-read",
    "user-read-recently-played",
    "user-top-read",
  ].join(" ");

  const params = new URLSearchParams({
    client_id: clientId,
    response_type: "code",
    redirect_uri: redirectUri,
    scope,
    code_challenge_method: "S256",
    code_challenge: challenge,
    state,
    show_dialog: "true",
  });

  const isProd = process.env.NODE_ENV === "production";
  const res = NextResponse.redirect("https://accounts.spotify.com/authorize?" + params.toString());
  // Also set one-time PKCE cookies (server-side read)
  res.cookies.set("pkce_verifier", verifier, { httpOnly: true, secure: isProd, sameSite: "lax", path: "/", maxAge: 600 });
  res.cookies.set("pkce_state", state,     { httpOnly: true, secure: isProd, sameSite: "lax", path: "/", maxAge: 600 });
  return res;
}
