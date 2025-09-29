"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export default function CallbackClient() {
  const params = useSearchParams();
  const router = useRouter();
  const [message, setMessage] = useState("Exchanging code...");
  const didRun = useRef(false);

  useEffect(() => {
    if (didRun.current) return;
    didRun.current = true;

    const code = params.get("code") ?? "";
    const state = params.get("state") ?? "";
    if (!code) {
      setMessage("No authorization code provided.");
      return;
    }

    fetch("/api/auth/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, state }),
    })
      .then(async (response) => {
        const payload = await response.text();
        if (!response.ok) throw new Error(payload);
        return JSON.parse(payload);
      })
      .then(() => {
        setMessage("Signed in! Redirecting...");
        router.replace("/");
      })
      .catch((error) => setMessage(`Auth failed: ${error instanceof Error ? error.message : String(error)}`));
  }, [params, router]);

  return <main style={{ padding: 24 }}>{message}</main>;
}