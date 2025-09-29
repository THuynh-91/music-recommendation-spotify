import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { applySpotifyCookies, refreshSpotifyTokens } from "@/lib/server-auth";

type RefreshPayload = {
  refresh_token?: string;
};

export async function POST(req: Request) {
  try {
    const body = (await req.json().catch(() => ({}))) as RefreshPayload;
    const cookieStore = cookies();
    const rt = body.refresh_token || cookieStore.get("sp_refresh_token")?.value;

    if (!rt) {
      return NextResponse.json({ error: "missing_refresh_token" }, { status: 400 });
    }

    const tokens = await refreshSpotifyTokens(rt);
    const res = NextResponse.json({ ok: true, expires_in: tokens.expires_in });
    applySpotifyCookies(res, tokens);
    return res;
  } catch (error) {
    const message = error instanceof Error ? error.message : "server_error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
