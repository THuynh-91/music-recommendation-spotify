import { NextResponse } from "next/server";

import { buildRealRecommendations, RecommenderError } from "@/lib/recommender";

interface RecommendRequestBody {
  url?: string;
  limit?: number;
  market?: string;
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
        : 15;

    // REAL, NO-AUTH PATH (primary):
    // Spotify deprecated its audio-features / recommendations endpoints in
    // Nov 2024, so the old credentialed backend can no longer return real
    // recommendations. Rather than fabricate songs, we resolve the pasted
    // track via Spotify's PUBLIC oEmbed endpoint and pull REAL similar tracks
    // from Deezer's public (no-key) API (related artists + their top tracks).
    // This needs no user credentials and returns genuine, relevant songs.
    try {
      const data = await buildRealRecommendations(payload.url, limit);
      return NextResponse.json(data);
    } catch (err) {
      const message =
        err instanceof RecommenderError
          ? err.message
          : err instanceof Error
            ? err.message
            : "recommender_failed";
      return NextResponse.json({ error: message }, { status: 502 });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "server_error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
