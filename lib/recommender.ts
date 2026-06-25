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

type DeezerAlbum = {
  cover?: string;
  cover_small?: string;
  cover_medium?: string;
  cover_big?: string;
  cover_xl?: string;
};

type DeezerTrack = {
  title?: string;
  title_short?: string;
  link?: string;
  artist?: { name?: string };
  album?: DeezerAlbum;
};

/** Pick the best available Deezer album cover URL for a card-sized image. */
function albumCover(album?: DeezerAlbum): string | null {
  if (!album) return null;
  return (
    album.cover_medium ||
    album.cover_big ||
    album.cover ||
    album.cover_small ||
    album.cover_xl ||
    null
  );
}

type DeezerSearchResponse = {
  data?: Array<
    DeezerTrack & {
      artist?: { id?: number; name?: string };
    }
  >;
  error?: { message?: string };
};

type DeezerSearchTrack = DeezerTrack & { artist?: { id?: number; name?: string } };

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

async function fetchText(url: string): Promise<string> {
  // IMPORTANT: Spotify serves two different pages for a track URL:
  //   - a JS Web Player shell (generic <title>, NO song-specific og: tags), and
  //   - an SEO/crawler page that DOES include og:title / og:description / og:image.
  // A normal browser UA + `Accept: text/html` gets the shell (no og: tags from
  // server-side fetch), which is exactly what broke disambiguation. Using a
  // crawler User-Agent and NOT sending the browser Accept header reliably
  // returns the SEO page with the meta tags we need to parse the real artist.
  const res = await fetch(url, {
    cache: "no-store",
    headers: {
      "User-Agent": "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
    },
  });
  if (!res.ok) {
    throw new RecommenderError(`Upstream request failed (${res.status}) for ${url}`);
  }
  return await res.text();
}

/** Pull the content of the first matching <meta property="<prop>" ...> tag. */
function metaContent(html: string, prop: string): string | undefined {
  // Be tolerant of attribute order: property/content can appear in either order.
  const patterns = [
    new RegExp(`<meta[^>]+property=["']${prop}["'][^>]+content=["']([^"']*)["']`, "i"),
    new RegExp(`<meta[^>]+content=["']([^"']*)["'][^>]+property=["']${prop}["']`, "i"),
  ];
  for (const re of patterns) {
    const m = html.match(re);
    if (m?.[1]) return decodeHtml(m[1].trim());
  }
  return undefined;
}

function decodeHtml(s: string): string {
  return s
    .replace(/&amp;/g, "&")
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/gi, "'")
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

/**
 * Resolve the REAL {title, artist, image} for a Spotify track by parsing its
 * public track page. The oEmbed endpoint only returns the title, so a track
 * that shares a name with another artist's song would resolve to the wrong
 * artist on Deezer. The track page exposes the artist via several meta tags;
 * we try them all and fall back gracefully. Returns undefined if nothing usable
 * could be parsed (callers then fall back to oEmbed).
 */
async function resolveFromTrackPage(url: string): Promise<
  | {
      title: string | undefined;
      artist: string | undefined;
      image: string | undefined;
    }
  | undefined
> {
  let html: string;
  try {
    html = await fetchText(url);
  } catch {
    return undefined;
  }

  const ogTitle = metaContent(html, "og:title");
  const ogDescription = metaContent(html, "og:description");
  const ogImage = metaContent(html, "og:image");

  let title = ogTitle || undefined;
  let artist: string | undefined;

  // og:description is typically "Artist · Album · Song · Year" -> artist is first.
  if (ogDescription) {
    const first = ogDescription.split("·")[0]?.trim();
    if (first && first.toLowerCase() !== "song") artist = first;
  }

  // Fall back to the <title>: "Song - song and lyrics by Artist | Spotify".
  if (!artist || !title) {
    const titleTag = html.match(/<title>([^<]*)<\/title>/i)?.[1];
    if (titleTag) {
      const decoded = decodeHtml(titleTag.trim());
      const byMatch = decoded.match(/^(.*?)\s+-\s+song and lyrics by\s+(.+?)\s*\|\s*Spotify/i);
      if (byMatch) {
        title = title || byMatch[1]?.trim();
        artist = artist || byMatch[2]?.trim();
      }
    }
  }

  if (!title && !artist && !ogImage) return undefined;
  return { title, artist, image: ogImage };
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
  // 1) Primary: parse the public track page for the REAL artist + title. This is
  //    what disambiguates songs that share a title across different artists.
  const page = await resolveFromTrackPage(url);

  // 2) oEmbed as a fallback for title/thumbnail (and if the page parse failed).
  let oembed: OEmbedResponse | undefined;
  try {
    oembed = await fetchJson<OEmbedResponse>(
      `${OEMBED_ENDPOINT}?url=${encodeURIComponent(url)}`,
    );
  } catch {
    oembed = undefined;
  }

  const oembedParsed = oembed ? parseOEmbedTitle(oembed) : { query: "", artistHint: undefined };

  const title = (page?.title || oembedParsed.query || "").trim();
  // The real artist comes from the track page; oEmbed author_name is a weak hint.
  const realArtist = (page?.artist || oembedParsed.artistHint || "").trim();
  const seedImage = page?.image ?? oembed?.thumbnail_url ?? null;

  if (!title) {
    throw new RecommenderError(
      "Could not read a title for this Spotify track (is the URL a public track?).",
    );
  }

  // 3) Scope the Deezer search by artist + track so we match the RIGHT song,
  //    not a same-named track by a different artist. Prefer a result whose
  //    artist actually matches the real artist; fall back progressively.
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  let best: DeezerSearchTrack | undefined;

  if (realArtist) {
    const scopedQuery = `artist:"${realArtist}" track:"${title}"`;
    try {
      const scoped = await fetchJson<DeezerSearchResponse>(
        `${DEEZER_API}/search?q=${encodeURIComponent(scopedQuery)}&limit=10`,
      );
      const candidates = (scoped.data ?? []).filter((t) => t.artist?.id);
      best =
        candidates.find((t) => t.artist?.name && norm(t.artist.name) === norm(realArtist)) ||
        candidates.find(
          (t) =>
            t.artist?.name &&
            (norm(t.artist.name).includes(norm(realArtist)) ||
              norm(realArtist).includes(norm(t.artist.name))),
        ) ||
        candidates[0];
    } catch {
      best = undefined;
    }
  }

  // Fallback A: plain "title artist" search, preferring an artist match.
  if (!best) {
    const searchQuery = realArtist ? `${title} ${realArtist}` : title;
    try {
      const search = await fetchJson<DeezerSearchResponse>(
        `${DEEZER_API}/search?q=${encodeURIComponent(searchQuery)}&limit=10`,
      );
      const candidates = (search.data ?? []).filter((t) => t.artist?.id);
      best = realArtist
        ? candidates.find((t) => t.artist?.name && norm(t.artist.name) === norm(realArtist)) ||
          candidates[0]
        : candidates[0];
    } catch {
      best = undefined;
    }
  }

  // Fallback B: bare title (original behaviour) so we never hard-error.
  if (!best) {
    const retry = await fetchJson<DeezerSearchResponse>(
      `${DEEZER_API}/search?q=${encodeURIComponent(title)}&limit=5`,
    );
    best = retry.data?.find((t) => t.artist?.id);
  }

  if (!best || !best.artist?.id) {
    throw new RecommenderError(
      `Resolved Spotify track "${title}"${
        realArtist ? ` by ${realArtist}` : ""
      } but could not match it to an artist on Deezer.`,
    );
  }

  return {
    // Trust the real artist from the track page when we have it; the Deezer
    // artist id drives related-artist recs, so it must be the right artist.
    title: best.title ?? title,
    artist: realArtist || best.artist.name || "Unknown artist",
    artistId: best.artist.id,
    image: seedImage ?? albumCover(best.album) ?? null,
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
      image_url: albumCover(track.album),
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
