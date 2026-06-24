"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";

import { fetchProfile, requestRecommendations } from "@/lib/spotify";
import type { RecommendationItem, RecommendationResponse, SpotifyProfile } from "@/lib/spotify";

const MAX_HISTORY = 5;

function formatArtists(artists: string[]) {
  return artists.join(", ");
}

/** Deterministic hue from a string so each placeholder cover is distinct but stable. */
function hueFromString(value: string) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash) % 360;
}

function initialsFrom(value: string) {
  return value
    .split(/\s+/)
    .map((part) => part.charAt(0))
    .filter(Boolean)
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

/**
 * Renders real Spotify artwork when available, otherwise an attractive
 * generative gradient placeholder so the (image-less) demo still looks polished.
 */
function Artwork({
  url,
  alt,
  variant = "thumb",
}: {
  url?: string | null;
  alt: string;
  variant?: "thumb" | "cover";
}) {
  const className = variant === "cover" ? "card-cover" : "artwork";
  if (url) {
    const size = variant === "cover" ? 480 : 112;
    return <Image src={url} alt={alt} width={size} height={size} className={className} />;
  }
  const hue = hueFromString(alt);
  const style = {
    background: `linear-gradient(135deg, hsl(${hue} 62% 32%), hsl(${(hue + 48) % 360} 58% 18%))`,
  };
  return (
    <div className={`${className} placeholder`} style={style} role="img" aria-label={alt}>
      <span className="placeholder-mark" aria-hidden="true">
        {initialsFrom(alt) || "♫"}
      </span>
      <svg className="placeholder-note" viewBox="0 0 24 24" aria-hidden="true">
        <path
          fill="currentColor"
          d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6Z"
        />
      </svg>
    </div>
  );
}

function ResultsSkeleton({ count }: { count: number }) {
  return (
    <section className="results" aria-busy="true" aria-label="Loading recommendations">
      <div className="results-header skeleton-header">
        <div className="skeleton skeleton-line skeleton-eyebrow" />
        <div className="skeleton skeleton-line skeleton-title" />
      </div>
      <div className="grid">
        {Array.from({ length: count }).map((_, idx) => (
          <article className="card skeleton-card" key={idx}>
            <div className="card-cover skeleton" />
            <div className="card-body">
              <div className="skeleton skeleton-line" style={{ width: "70%" }} />
              <div className="skeleton skeleton-line" style={{ width: "45%" }} />
              <div className="skeleton skeleton-line" style={{ width: "90%" }} />
              <div className="skeleton skeleton-line" style={{ width: "30%", height: 22 }} />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export function HomeClient({ signedIn, demoMode = false }: { signedIn: boolean; demoMode?: boolean }) {
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
    return initialsFrom(profile.display_name);
  }, [profile]);

  const runExample = (url: string) => {
    setInputUrl(url);
    void submitUrl(url);
  };

  const submitUrl = async (rawUrl: string) => {
    const url = rawUrl.trim();
    if (!url) {
      setError("Please paste a Spotify track or playlist URL");
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const data = await requestRecommendations(url, limit);
      setResult(data);
      setHistory((prev) => [data, ...prev].slice(0, MAX_HISTORY));
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unable to fetch recommendations";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void submitUrl(inputUrl);
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
          <span className="hero-eyebrow">Spotify Recommendation Studio</span>
          <h1>Find your next favourite track</h1>
          <p>
            Connect Spotify and drop any track or playlist link. We analyse audio features and
            similarity to surface recommendations, each with a human-readable reason why.
          </p>
          <button className="cta cta-spotify" onClick={handleSignIn}>
            <SpotifyGlyph />
            Connect Spotify
          </button>
          <p className="hero-foot subtle">No Spotify app configured? Launch with demo data to explore the UI.</p>
        </div>
      </section>
    );
  }

  return (
    <div className="dashboard">
      {demoMode && (
        <div className="demo-banner" role="status">
          <span className="demo-pill">Demo</span>
          <span className="demo-text">
            You are exploring with sample data; Spotify is not connected. Add your Spotify
            credentials to <code>.env</code> to enable the live flow.
          </span>
        </div>
      )}
      <header className="dashboard-header">
        <div className="profile">
          {profile?.images?.[0]?.url ? (
            <Artwork url={profile.images[0].url} alt={profile.display_name ?? profile.id} />
          ) : (
            <div className="avatar-fallback">{profileInitials || "?"}</div>
          )}
          <div>
            <p className="profile-name">{profile?.display_name ?? profile?.id ?? "Spotify user"}</p>
            {profileLoading ? (
              <span className="profile-subtle">Refreshing profile...</span>
            ) : profileError ? (
              <span className="profile-error">{profileError}</span>
            ) : (
              <span className="profile-subtle">{demoMode ? "Demo session" : "Connected"}</span>
            )}
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
        <div className="examples">
          <span className="examples-label">Try an example:</span>
          <button
            type="button"
            className="chip"
            disabled={loading}
            onClick={() => runExample("https://open.spotify.com/track/demo-seed")}
          >
            Sample track
          </button>
          <button
            type="button"
            className="chip"
            disabled={loading}
            onClick={() => runExample("https://open.spotify.com/playlist/demo-seed")}
          >
            Sample playlist
          </button>
        </div>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </section>

      {loading && <ResultsSkeleton count={Math.min(limit, 6)} />}

      {!loading && !result && (
        <section className="empty-state">
          <div className="empty-art" aria-hidden="true">♫</div>
          <h3>No recommendations yet</h3>
          <p className="subtle">
            Paste a Spotify track or playlist link above, or tap an example to see how the
            recommender explains each match.
          </p>
        </section>
      )}

      {!loading && result && (
        <section className="results">
          <header className="results-header">
            <div>
              <p className="eyebrow">Results</p>
              <h3>{result.type === "track" ? "Based on this track" : "Based on this playlist"}</h3>
            </div>
            {result.seed_track && (
              <div className="seed">
                <Artwork url={result.seed_track.image_url ?? null} alt={result.seed_track.name} />
                <div>
                  <p className="seed-title">{result.seed_track.name}</p>
                  <p className="seed-subtitle">{formatArtists(result.seed_track.artists)}</p>
                </div>
              </div>
            )}
            {result.seed_playlist && (
              <div className="seed">
                <Artwork url={result.seed_playlist.image_url ?? null} alt={result.seed_playlist.name} />
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

          <p className="results-count subtle">
            {result.recommendations.length} recommendation
            {result.recommendations.length === 1 ? "" : "s"}, ranked by similarity
          </p>
          <div className="grid">
            {result.recommendations.map((item, idx) => (
              <RecommendationCard key={item.track_id} item={item} rank={idx + 1} />
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

function RecommendationCard({ item, rank }: { item: RecommendationItem; rank: number }) {
  const match = Math.round(item.similarity * 100);
  return (
    <article className="card">
      <div className="card-cover-wrap">
        <Artwork url={item.image_url ?? null} alt={item.name} variant="cover" />
        <span className="card-rank">#{rank}</span>
      </div>
      <div className="card-body">
        <div>
          <h4 title={item.name}>{item.name}</h4>
          <p className="subtle">{formatArtists(item.artists)}</p>
        </div>
        <div className="match" aria-label={`Match ${match} percent`}>
          <div className="match-bar">
            <span className="match-fill" style={{ width: `${match}%` }} />
          </div>
          <span className="badge">{match}% match</span>
        </div>
        <p className="explanation">{item.explanation}</p>
        <footer>
          {item.external_url && (
            <a
              href={item.external_url}
              target="_blank"
              rel="noreferrer"
              className="ghost ghost-spotify"
            >
              <SpotifyGlyph />
              Open in Spotify
            </a>
          )}
        </footer>
      </div>
    </article>
  );
}

function SpotifyGlyph() {
  return (
    <svg className="spotify-glyph" viewBox="0 0 24 24" aria-hidden="true" width="16" height="16">
      <path
        fill="currentColor"
        d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm4.59 14.43a.62.62 0 0 1-.86.21c-2.35-1.44-5.3-1.76-8.79-.97a.62.62 0 1 1-.28-1.21c3.82-.87 7.1-.49 9.72 1.11.3.18.39.57.21.86Zm1.22-2.72a.78.78 0 0 1-1.07.26c-2.69-1.65-6.79-2.13-9.97-1.17a.78.78 0 1 1-.45-1.49c3.64-1.1 8.16-.56 11.24 1.33.37.22.49.7.25 1.07Zm.11-2.84C14.8 8.96 9.5 8.78 6.43 9.71a.93.93 0 1 1-.54-1.78c3.53-1.07 9.39-.86 13.09 1.34a.93.93 0 1 1-.95 1.6Z"
      />
    </svg>
  );
}
