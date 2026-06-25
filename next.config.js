/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      { protocol: 'https', hostname: 'i.scdn.co' },
      { protocol: 'https', hostname: 'mosaic.scdn.co' },
      // Spotify oEmbed thumbnails (seed track artwork) are served from these hosts.
      { protocol: 'https', hostname: '*.spotifycdn.com' },
      { protocol: 'https', hostname: '*.scdn.co' },
      // Deezer album covers (recommendation card artwork) come from these hosts.
      { protocol: 'https', hostname: 'cdn-images.dzcdn.net' },
      { protocol: 'https', hostname: '*.dzcdn.net' },
      { protocol: 'https', hostname: 'api.deezer.com' },
      { protocol: 'https', hostname: 'e-cdns-images.dzcdn.net' },
    ],
  },
  output: 'standalone',
};

module.exports = nextConfig;