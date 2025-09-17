"use client";
export default function Page() {
  const signIn = () => { window.location.href = "/api/auth/start"; };
  return (
    <main style={{ padding: 16 }}>
      <button onClick={signIn} style={{ padding: 10, border: "1px solid #ccc", borderRadius: 8 }}>
        Sign in with Spotify
      </button>
    </main>
  );
}