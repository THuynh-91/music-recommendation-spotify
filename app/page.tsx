import { cookies } from "next/headers";

import { HomeClient } from "./components/HomeClient";

export const dynamic = "force-dynamic";

export default function Page() {
  const cookieStore = cookies();
  // The recommender needs NO Spotify login: it resolves tracks via Spotify's
  // public oEmbed and pulls real similar songs from Deezer's public API. So the
  // dashboard is always available. We surface a "no-auth (Deezer)" banner unless
  // the user happens to have a live Spotify session.
  const hasSpotifySession = Boolean(
    cookieStore.get("sp_access_token") || cookieStore.get("sp_refresh_token"),
  );
  const noAuthMode = !hasSpotifySession;

  return (
    <main className="page">
      <HomeClient signedIn noAuthMode={noAuthMode} />
    </main>
  );
}
