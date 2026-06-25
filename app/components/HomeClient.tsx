"use client";

import { useEffect, useMemo, useRef, useState } from "react";

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

function ArtworkPlaceholder({ alt, className }: { alt: string; className: string }) {
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

/**
 * Renders real cover artwork when available, otherwise an attractive generative
 * gradient placeholder so the experience still looks polished.
 *
 * The artwork comes from arbitrary external CDNs (Spotify oEmbed thumbnails,
 * Deezer album covers whose hosts rotate). We deliberately use a plain <img>
 * instead of next/image: next/image throws a render-time error ("Invalid src
 * prop ... hostname is not configured under images") for any host missing from
 * next.config.js remotePatterns, and with no error boundary that error blanks
 * the entire page. A plain <img> with an onError fallback can never crash the
 * render — a broken or unconfigured cover simply falls back to the placeholder.
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
  const [failed, setFailed] = useState(false);
  // Reset the failure flag if the URL changes (e.g. a card is reused).
  useEffect(() => setFailed(false), [url]);

  if (url && !failed) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={url}
        alt={alt}
        className={className}
        loading="lazy"
        decoding="async"
        referrerPolicy="no-referrer"
        onError={() => setFailed(true)}
      />
    );
  }
  return <ArtworkPlaceholder alt={alt} className={className} />;
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

export function HomeClient({ signedIn, noAuthMode = false }: { signedIn: boolean; noAuthMode?: boolean }) {
  const [profile, setProfile] = useState<SpotifyProfile | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);

  const [inputUrl, setInputUrl] = useState("");
  const [limit, setLimit] = useState(15);
  // Raw text backing the count <input> so the user can freely clear and retype.
  // We clamp to a valid number only on blur; `limit` stays the numeric source of
  // truth used for the actual request.
  const [limitText, setLimitText] = useState("15");
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

  // Auto-load a default example on first mount so visitors land on real
  // recommendations instead of an empty form. Runs once.
  const didAutoload = useRef(false);
  useEffect(() => {
    if (didAutoload.current) return;
    didAutoload.current = true;
    const DEFAULT_SEED = "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b"; // Blinding Lights — The Weeknd
    setInputUrl(DEFAULT_SEED);
    void submitUrl(DEFAULT_SEED);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
            Drop any Spotify track link and get real similar songs - no login required. We resolve
            the track via Spotify&apos;s public oEmbed and pull related tracks from Deezer&apos;s
            public API.
          </p>
          <button className="cta cta-spotify" onClick={handleSignIn}>
            <SpotifyGlyph />
            Connect Spotify
          </button>
          <p className="hero-foot subtle">No login needed - recommendations come from Deezer&apos;s public catalogue.</p>
        </div>
      </section>
    );
  }

  return (
    <div className="dashboard">
      {noAuthMode && (
        <div className="demo-banner" role="status">
          <span className="demo-pill">Live</span>
          <span className="demo-text">
            Real recommendations, no Spotify login required. Tracks are resolved via Spotify&apos;s
            public oEmbed and similar songs come from <strong>Deezer&apos;s public API</strong>
            (related artists). These are real songs, not samples.
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
              <span className="profile-subtle">{noAuthMode ? "No-auth mode (Deezer)" : "Connected"}</span>
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
            value={limitText}
            onChange={(event) => {
              const raw = event.target.value;
              // Let the field be freely edited (including temporarily empty).
              setLimitText(raw);
              const parsed = Number(raw);
              if (raw !== "" && Number.isFinite(parsed)) {
                setLimit(Math.min(Math.max(Math.trunc(parsed), 1), 50));
              }
            }}
            onBlur={() => {
              // Normalise the visible value once editing finishes.
              const parsed = Number(limitText);
              const clamped =
                limitText === "" || !Number.isFinite(parsed)
                  ? limit
                  : Math.min(Math.max(Math.trunc(parsed), 1), 50);
              setLimit(clamped);
              setLimitText(String(clamped));
            }}
            disabled={loading}
          />
          <button className="cta" type="submit" disabled={loading}>
            {loading ? "Finding similar songs..." : "Get recommendations"}
          </button>
        </form>
        <div className="examples">
          <span className="examples-label">Try an example:</span>
          <button
            type="button"
            className="chip"
            disabled={loading}
            onClick={() => runExample("https://open.spotify.com/track/32lItqlMi4LBhb4k0BaSaC")}
          >
            Candy Paint - Post Malone
          </button>
          <button
            type="button"
            className="chip"
            disabled={loading}
            onClick={() => runExample("https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b")}
          >
            Blinding Lights - The Weeknd
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
            {result.recommendations.length} real recommendation
            {result.recommendations.length === 1 ? "" : "s"} via Deezer related artists
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
  const isSpotify = item.external_url?.includes("open.spotify.com");
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
              {isSpotify ? "Open in Spotify" : "Open on Deezer"}
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
