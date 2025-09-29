import { Suspense } from "react";

import CallbackClient from "./callback-client";

export const dynamic = "force-dynamic";

export default function CallbackPage() {
  return (
    <Suspense fallback={<main style={{ padding: 24 }}>Processing Spotify callback...</main>}>
      <CallbackClient />
    </Suspense>
  );
}