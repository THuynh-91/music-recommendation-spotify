import type { RecommendationResponse, SpotifyProfile } from "@/lib/spotify";

/**
 * Demo / degraded-mode fixtures so the UI is fully exercisable without a real
 * Spotify app or running recommender backend. Activated when SPOTIFY_CLIENT_ID
 * / RECOMMENDER_SERVICE_TOKEN are absent, or DEMO_MODE=1.
 */

export const DEMO_PROFILE: SpotifyProfile = {
  id: "demo-user",
  display_name: "Demo Listener",
  email: "demo@example.com",
  images: [],
};

const DEMO_TRACKS = [
  { name: "Midnight City", artists: ["M83"], explanation: "High energy; matching tempo around 105 BPM" },
  { name: "Instant Crush", artists: ["Daft Punk", "Julian Casablancas"], explanation: "Similar danceability; shares electronic vibes" },
  { name: "Electric Feel", artists: ["MGMT"], explanation: "Similar mood; high energy" },
  { name: "Tighten Up", artists: ["The Black Keys"], explanation: "Matching tempo; shares rock vibes" },
  { name: "Dog Days Are Over", artists: ["Florence + The Machine"], explanation: "High valence; similar energy" },
  { name: "Take Me Out", artists: ["Franz Ferdinand"], explanation: "Matching tempo; high danceability" },
  { name: "Feel It Still", artists: ["Portugal. The Man"], explanation: "Similar mood; shares indie vibes" },
  { name: "Pumped Up Kicks", artists: ["Foster The People"], explanation: "Similar danceability; matching tempo" },
];

export function buildDemoRecommendations(url: string, limit: number): RecommendationResponse {
  const isPlaylist = url.includes("playlist");
  const count = Math.min(Math.max(limit, 1), DEMO_TRACKS.length);
  const recommendations = DEMO_TRACKS.slice(0, count).map((t, idx) => ({
    track_id: `demo-${idx}`,
    name: t.name,
    artists: t.artists,
    preview_url: null,
    external_url: "https://open.spotify.com/",
    image_url: null,
    similarity: Math.max(0.6, 0.95 - idx * 0.04),
    explanation: `${t.explanation} (demo data)`,
  }));

  if (isPlaylist) {
    return {
      type: "playlist",
      seed_playlist: {
        id: "demo-playlist",
        name: "Demo Playlist",
        description: "Sample playlist (demo mode)",
        owner: "Demo Listener",
        image_url: null,
        track_count: 25,
      },
      playlist: {
        id: "demo-playlist",
        name: "Demo Playlist",
        track_count: 25,
        ingested_tracks: 25,
        snapshot_id: "demo-snapshot",
      },
      recommendations,
    };
  }

  return {
    type: "track",
    seed_track: {
      id: "demo-seed",
      name: "Demo Seed Track",
      artists: ["Demo Artist"],
      image_url: null,
      external_url: "https://open.spotify.com/",
    },
    recommendations,
  };
}
