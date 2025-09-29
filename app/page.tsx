import { cookies } from "next/headers";

import { HomeClient } from "./components/HomeClient";

export const dynamic = "force-dynamic";

export default function Page() {
  const cookieStore = cookies();
  const signedIn = Boolean(cookieStore.get("sp_access_token") || cookieStore.get("sp_refresh_token"));

  return (
    <main className="page">
      <HomeClient signedIn={signedIn} />
    </main>
  );
}