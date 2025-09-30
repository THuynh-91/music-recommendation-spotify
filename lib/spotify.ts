export type RecommendationItem = {
  track_id: string;
  name: string;
  artists: string[];
  preview_url?: string | null;
  external_url?: string | null;
  image_url?: string | null;
  similarity: number;
  explanation: string;
};

export type SeedTrack = {
  id: string;
  name: string;
  artists: string[];
  image_url?: string | null;
  external_url?: string | null;
};

export type SeedPlaylist = {
  id: string;
  name: string;
  description?: string | null;
  owner?: string | null;
  image_url?: string | null;
  track_count: number;
};

export type PlaylistIngestSummary = {
  id: string;
  name: string;
  track_count: number;
  ingested_tracks: number;
  snapshot_id?: string | null;
};

export type RecommendationResponse = {
  type: "track" | "playlist";
  seed_track?: SeedTrack | null;
  seed_playlist?: SeedPlaylist | null;
  playlist?: PlaylistIngestSummary | null;
  recommendations: RecommendationItem[];
};

export type SpotifyProfile = {
  id: string;
  display_name?: string;
  images?: { url: string }[];
  email?: string;
};

export async function requestRecommendations(url: string, limit?: number): Promise<RecommendationResponse> {
  const payload: { url: string; limit?: number } = { url };
  if (typeof limit === "number") {
    payload.limit = limit;
  }
  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.error ?? "Failed to fetch recommendations");
  }
  return data as RecommendationResponse;
}

export async function fetchProfile(): Promise<SpotifyProfile> {
  const res = await fetch("/api/auth/profile", { cache: "no-store" });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.error ?? "Failed to load profile");
  }
  return data as SpotifyProfile;
}