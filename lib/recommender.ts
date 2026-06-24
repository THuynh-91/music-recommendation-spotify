import type { RecommendationItem, RecommendationResponse } from "@/lib/spotify";

/**
 * Real, no-auth recommender.
 *
 * Pipeline (no user credentials, no Spotify app required):
 *   1. Resolve the pasted Spotify track URL -> real title via Spotify's PUBLIC
 *      oEmbed endpoint (https://open.spotify.com/oembed).
 *   2. Resolve title -> {artist, deezer artist id} via Deezer's public search API.
 *   3. Pull REAL similar tracks from Deezer: related artists' top tracks plus the
 *      seed artist's own top tracks. Every item returned is a real Deezer track.
 *
 * Spotify deprecated audio-features / recommendations (Nov 2024), so there is no
 * "audio feature" similarity here and we do not pretend there is. Relevance is
 * Deezer's editorial "related artists" graph, which is genuine collaborative data.
 */

const OEMBED_ENDPOINT = "https://open.spotify.com/oembed";
const DEEZER_API = "https://api.deezer.com";

export class RecommenderError extends Error {}

type DeezerTrack = {
  title?: string;
  title_short?: string;
  link?: string;
  artist?: { name?: string };
};

type DeezerSearchResponse = {
  data?: Array<
    DeezerTrack & {
      artist?: { id?: number; name?: string };
    }
  >;
  error?: { message?: string };
};

type DeezerRelatedResponse = {
  data?: Array<{ id?: number; name?: string }>;
  error?: { message?: string };
};

type DeezerTopResponse = {
  data?: DeezerTrack[];
  error?: { message?: string };
};

type OEmbedResponse = {
  title?: string;
  author_name?: string;
  thumbnail_url?: string;
};

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    cache: "no-store",
    headers: { "User-Agent": "spotify-rec/1.0 (+https://localhost)" },
  });
  if (!res.ok) {
    throw new RecommenderError(`Upstream request failed (${res.status}) for ${url}`);
  }
  return (await res.json()) as T;
}

function spotifySearchLink(title: string, artist: string): string {
  return `https://open.spotify.com/search/${encodeURIComponent(`${title} ${artist}`)}`;
}

/**
 * Parse the oEmbed title. Spotify's oEmbed sometimes returns just the song
 * title, sometimes "Artist - Song". We keep the raw title for searching and
 * use author_name when present as an artist hint.
 */
function parseOEmbedTitle(oembed: OEmbedResponse): { query: string; artistHint: string | undefined } {
  const title = (oembed.title ?? "").trim();
  const artistHint = oembed.author_name?.trim() || undefined;
  return { query: title, artistHint };
}

/**
 * Resolve a pasted Spotify track URL to a real seed track using public oEmbed +
 * Deezer search. Throws RecommenderError if it cannot be resolved.
 */
async function resolveSeed(url: string): Promise<{
  title: string;
  artist: string;
  artistId: number;
  image: string | null;
}> {
  const oembedUrl = `${OEMBED_ENDPOINT}?url=${encodeURIComponent(url)}`;
  let oembed: OEmbedResponse;
  try {
    oembed = await fetchJson<OEmbedResponse>(oembedUrl);
  } catch (err) {
    throw new RecommenderError(
      `Could not read the Spotify track via oEmbed (is the URL a public track?): ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
  }

  const { query, artistHint } = parseOEmbedTitle(oembed);
  if (!query) {
    throw new RecommenderError("Spotify oEmbed returned no title for this URL.");
  }

  // Search Deezer for the title (plus artist hint if available) to get the real
  // artist + Deezer artist id needed for related-artist lookups.
  const searchQuery = artistHint ? `${query} ${artistHint}` : query;
  const search = await fetchJson<DeezerSearchResponse>(
    `${DEEZER_API}/search?q=${encodeURIComponent(searchQuery)}&limit=5`,
  );
  let best = search.data?.find((t) => t.artist?.id);
  if (!best) {
    // Retry with the bare title in case the artist hint hurt the match.
    const retry = await fetchJson<DeezerSearchResponse>(
      `${DEEZER_API}/search?q=${encodeURIComponent(query)}&limit=5`,
    );
    best = retry.data?.find((t) => t.artist?.id);
  }
  if (!best || !best.artist?.id) {
    throw new RecommenderError(
      `Resolved Spotify title "${query}" but could not match it to an artist on Deezer.`,
    );
  }

  return {
    title: best.title ?? query,
    artist: best.artist.name ?? artistHint ?? "Unknown artist",
    artistId: best.artist.id,
    image: oembed.thumbnail_url ?? null,
  };
}

/**
 * Build EXACTLY `limit` real recommendations from Deezer related artists.
 * Returns fewer only if Deezer genuinely has no more related tracks.
 */
export async function buildRealRecommendations(
  url: string,
  limit: number,
): Promise<RecommendationResponse> {
  const target = Math.min(Math.max(Math.trunc(limit), 1), 50);
  const seed = await resolveSeed(url);

  const related = await fetchJson<DeezerRelatedResponse>(
    `${DEEZER_API}/artist/${seed.artistId}/related?limit=25`,
  );
  const relatedArtists = (related.data ?? []).filter((a) => a.id);

  const seen = new Set<string>();
  const items: RecommendationItem[] = [];

  const addTrack = (track: DeezerTrack, reason: string) => {
    const name = track.title ?? track.title_short;
    const artist = track.artist?.name;
    if (!name || !artist) return;
    const key = `${name.toLowerCase()}::${artist.toLowerCase()}`;
    if (seen.has(key)) return;
    // Skip the exact seed track itself.
    if (
      name.toLowerCase() === seed.title.toLowerCase() &&
      artist.toLowerCase() === seed.artist.toLowerCase()
    ) {
      return;
    }
    seen.add(key);
    items.push({
      track_id: `dz-${items.length}`,
      name,
      artists: [artist],
      preview_url: null,
      external_url: track.link ?? spotifySearchLink(name, artist),
      image_url: null,
      similarity: 0,
      explanation: reason,
    });
  };

  // Pull a couple of top tracks from each related artist until we hit the target.
  // Pass 1: 2 tracks per related artist (breadth). Pass 2: fill from remaining.
  for (const pass of [2, 5]) {
    for (const artist of relatedArtists) {
      if (items.length >= target) break;
      let top: DeezerTopResponse;
      try {
        top = await fetchJson<DeezerTopResponse>(
          `${DEEZER_API}/artist/${artist.id}/top?limit=${pass}`,
        );
      } catch {
        continue;
      }
      for (const track of top.data ?? []) {
        if (items.length >= target) break;
        addTrack(track, `Related to ${seed.artist} (via ${artist.name})`);
      }
    }
    if (items.length >= target) break;
  }

  // Fallback: top up with the seed artist's own catalogue if related pool was thin.
  if (items.length < target) {
    try {
      const ownTop = await fetchJson<DeezerTopResponse>(
        `${DEEZER_API}/artist/${seed.artistId}/top?limit=50`,
      );
      for (const track of ownTop.data ?? []) {
        if (items.length >= target) break;
        addTrack(track, `More from ${seed.artist}`);
      }
    } catch {
      /* ignore */
    }
  }

  return {
    type: "track",
    seed_track: {
      id: `dz-seed-${seed.artistId}`,
      name: seed.title,
      artists: [seed.artist],
      image_url: seed.image,
      external_url: url,
    },
    recommendations: items.slice(0, target),
  };
}
