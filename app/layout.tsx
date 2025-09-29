import "./globals.css";

export const metadata = {
  title: "Spotify Recommendation Studio",
  description: "Analyze Spotify tracks and playlists with FAISS-powered intelligence",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}