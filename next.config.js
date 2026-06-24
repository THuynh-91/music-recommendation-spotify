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
    ],
  },
  output: 'standalone',
};

module.exports = nextConfig;