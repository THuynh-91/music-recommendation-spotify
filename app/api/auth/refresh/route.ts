import { NextResponse } from "next/server";
import { cookies } from "next/headers";

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({} as any));
    const cookieStore = cookies();
    const rt = body.refresh_token || cookieStore.get("sp_refresh_token")?.value;

    if (!rt) return NextResponse.json({ error: "Missing refresh_token" }, { status: 400 });

    const params = new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: rt,
      client_id: process.env.SPOTIFY_CLIENT_ID ?? "",
    });

    const r = await fetch("https://accounts.spotify.com/api/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: params.toString(),
    });

    const data = await r.json();
    if (!r.ok) return NextResponse.json(data, { status: r.status });

    const isProd = process.env.NODE_ENV === "production";
    const res = NextResponse.json({ ok: true, expires_in: data.expires_in });

    if (data.access_token) {
      res.cookies.set("sp_access_token", data.access_token, {
        httpOnly: true,
        secure: isProd,
        sameSite: "lax",
        path: "/",
        maxAge: data.expires_in ?? 3600,
      });
    }

    if (data.refresh_token) {
      res.cookies.set("sp_refresh_token", data.refresh_token, {
        httpOnly: true,
        secure: isProd,
        sameSite: "lax",
        path: "/",
        maxAge: 60 * 60 * 24 * 30,
      });
    }

    return res;
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "server_error" }, { status: 500 });
  }
}