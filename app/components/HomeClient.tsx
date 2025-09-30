"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";

import { fetchProfile, requestRecommendations } from "@/lib/spotify";
import type { RecommendationItem, RecommendationResponse, SpotifyProfile } from "@/lib/spotify";

const MAX_HISTORY = 5;

function formatArtists(artists: string[]) {
  return artists.join(", ");
}

function SpotifyImage({ url, alt, className }: { url?: string | null; alt: string; className?: string }) {
  if (!url) return null;
  return <Image src={url} alt={alt} width={240} height={240} className={className ?? "artwork"} />;
}

export function HomeClient({ signedIn }: { signedIn: boolean }) {
  const [profile, setProfile] = useState<SpotifyProfile | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);

  const [inputUrl, setInputUrl] = useState("");
  const [limit, setLimit] = useState(15);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [history, setHistory] = useState<RecommendationResponse[]>([]);

  useEffect(() => {
    if (!signedIn) {
      setProfile(null);
      setHistory([]);
      return;
    }
    setProfileLoading(true);
    fetchProfile()
      .then((data) => {
        setProfile(data);
        setProfileError(null);
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : "Unable to load profile";
        setProfileError(message);
      })
      .finally(() => setProfileLoading(false));
  }, [signedIn]);

  const profileInitials = useMemo(() => {
    if (!profile?.display_name) return "";
    return profile.display_name
      .split(" ")
      .map((part) => part.charAt(0))
      .join("")
      .slice(0, 2)
      .toUpperCase();
  }, [profile]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!inputUrl.trim()) {
      setError("Please paste a Spotify track or playlist URL");
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const data = await requestRecommendations(inputUrl.trim(), limit);
      setResult(data);
      setHistory((prev) => [data, ...prev].slice(0, MAX_HISTORY));
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unable to fetch recommendations";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleSignIn = () => {
    window.location.href = "/api/auth/start";
  };

  const handleSignOut = async () => {
    await fetch("/api/auth/signout", { method: "POST" });
    window.location.href = "/";
  };

  if (!signedIn) {
    return (
      <section className="hero">
        <div className="hero-content">
          <h1>Spotify Intelligence, Built For You</h1>
          <p>
            Sign in and drop any Spotify link. We analyse audio features, structural segments, and DSP descriptors to
            return recommendations powered by FAISS with human-readable explanations.
          </p>
          <button className="cta" onClick={handleSignIn}>
            Sign in with Spotify
          </button>
        </div>
      </section>
    );
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="profile">
          {profile?.images?.[0]?.url ? (
            <SpotifyImage url={profile.images[0].url} alt={profile.display_name ?? profile.id} className="artwork" />
          ) : (
            <div className="avatar-fallback">{profileInitials || "?"}</div>
          )}
          <div>
            <p className="profile-name">{profile?.display_name ?? profile?.id ?? "Spotify user"}</p>
            {profileLoading && <span className="profile-subtle">Refreshing profile...</span>}
            {profileError && <span className="profile-error">{profileError}</span>}
          </div>
        </div>
        <button className="ghost" onClick={handleSignOut}>
          Sign out
        </button>
      </header>

      <section className="panel">
        <h2>Analyze a track or playlist</h2>
        <form className="form" onSubmit={handleSubmit}>
          <label htmlFor="url">Spotify URL</label>
          <input
            id="url"
            type="url"
            placeholder="https://open.spotify.com/track/..."
            value={inputUrl}
            onChange={(event) => setInputUrl(event.target.value)}
            required
            disabled={loading}
          />
          <label htmlFor="limit">How many recommendations?</label>
          <input
            id="limit"
            type="number"
            min={1}
            max={50}
            value={limit}
            onChange={(event) => {
              const parsed = Number(event.target.value);
              if (!Number.isNaN(parsed)) {
                const clamped = Math.min(Math.max(Math.trunc(parsed), 1), 50);
                setLimit(clamped);
              }
            }}
            disabled={loading}
          />
          <button className="cta" type="submit" disabled={loading}>
            {loading ? "Crunching audio features..." : "Get recommendations"}
          </button>
        </form>
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="results">
          <header className="results-header">
            <div>
              <p className="eyebrow">Results</p>
              <h3>{result.type === "track" ? "Based on this track" : "Playlist vector summary"}</h3>
            </div>
            {result.seed_track && (
              <div className="seed">
                <SpotifyImage url={result.seed_track.image_url ?? null} alt={result.seed_track.name} className="artwork" />
                <div>
                  <p className="seed-title">{result.seed_track.name}</p>
                  <p className="seed-subtitle">{formatArtists(result.seed_track.artists)}</p>
                </div>
              </div>
            )}
            {result.seed_playlist && (
              <div className="seed">
                <SpotifyImage
                  url={result.seed_playlist.image_url ?? null}
                  alt={result.seed_playlist.name}
                  className="artwork"
                />
                <div>
                  <p className="seed-title">{result.seed_playlist.name}</p>
                  <p className="seed-subtitle">{result.seed_playlist.track_count} tracks curated</p>
                </div>
              </div>
            )}
          </header>

          {result.playlist && (
            <div className="playlist-summary">
              <p>This catalog now tracks {result.playlist.ingested_tracks} songs.</p>
              <p className="subtle">
                Snapshot {result.playlist.snapshot_id ?? "latest"}; total tracks: {result.playlist.track_count}
              </p>
            </div>
          )}

          <div className="grid">
            {result.recommendations.map((item) => (
              <RecommendationCard key={item.track_id} item={item} />
            ))}
          </div>
        </section>
      )}

      {history.length > 1 && (
        <section className="history">
          <h3>Recent runs</h3>
          <ul>
            {history.slice(1).map((entry, idx) => (
              <li key={`${entry.type}-${idx}`}>
                {entry.type === "track" ? entry.seed_track?.name : entry.seed_playlist?.name}
                <span className="subtle"> - {entry.recommendations.length} recs</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function RecommendationCard({ item }: { item: RecommendationItem }) {
  return (
    <article className="card">
      <SpotifyImage url={item.image_url ?? null} alt={item.name} className="card-cover" />
      <div className="card-body">
        <div>
          <h4>{item.name}</h4>
          <p className="subtle">{formatArtists(item.artists)}</p>
        </div>
        <p className="explanation">{item.explanation}</p>
        <footer>
          <span className="badge">Match {(item.similarity * 100).toFixed(0)}%</span>
          {item.external_url && (
            <a href={item.external_url} target="_blank" rel="noreferrer" className="ghost">
              Open in Spotify
            </a>
          )}
        </footer>
      </div>
    </article>
  );
}
