"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export default function Callback() {
  const params = useSearchParams();
  const router = useRouter();
  const [msg, setMsg] = useState("Exchanging code...");
  const didRun = useRef(false);   // <-- guard

  useEffect(() => {
    if (didRun.current) return;   // block duplicate dev/Strict runs
    didRun.current = true;

    const code = params.get("code") || "";
    const state = params.get("state") || "";
    if (!code) { setMsg("No authorization code provided."); return; }

    fetch("/api/auth/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, state }),
    })
      .then(async (r) => {
        const text = await r.text();
        if (!r.ok) throw new Error(text);
        return JSON.parse(text);
      })
      .then(() => { setMsg("Signed in! Redirecting..."); router.replace("/"); })
      .catch((e) => setMsg("Auth failed: " + e.message));
  }, [params, router]);

  return <main style={{ padding: 16 }}>{msg}</main>;
}