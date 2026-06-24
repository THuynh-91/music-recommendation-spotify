import { cookies } from "next/headers";

import { HomeClient } from "./components/HomeClient";

export const dynamic = "force-dynamic";

export default function Page() {
  const cookieStore = cookies();
  // Demo mode: no Spotify app configured. Let the user exercise the UI with
  // mock data instead of showing a dead "connect Spotify" wall.
  const demoMode = process.env.DEMO_MODE === "1" || !process.env.SPOTIFY_CLIENT_ID;
  const signedIn =
    demoMode ||
    Boolean(cookieStore.get("sp_access_token") || cookieStore.get("sp_refresh_token"));

  return (
    <main className="page">
      <HomeClient signedIn={signedIn} demoMode={demoMode} />
    </main>
  );
}
