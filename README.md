# 🎵 Music Recommendation with Spotify 

This is a full-stack web app that analyzes your Spotify tracks and playlists, then generates personalized music recommendations.  
The project combines **Next.js (frontend)** and **FastAPI (backend)**, with Docker for deployment. WIP

---

## 🚀 Features
- Input a **Spotify track or playlist URL**.
- Detect whether it’s a single track or playlist.
- For tracks:
  - Fetch **Spotify Audio Features** (tempo, energy, danceability, etc.).
  - Fetch **Audio Analysis** (beats, sections, time signature).
  - Optionally compute DSP features (via librosa).
- For playlists:
  - Build a **catalog of tracks** with extracted features.
  - Store features in a **FAISS vector index** for fast similarity search.
- Generate **Top-K recommendations** using nearest neighbors with genre soft-boost and diversity re-ranking.
- Each recommendation includes a clear, human-readable explanation of *why it was suggested*.
- Fully containerized with **Docker Compose**.

---

##Tech Stack
- **Frontend**: [Next.js](https://nextjs.org/) + TypeScript  
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) + Python 3.11  
- **ML / DSP**: librosa, FAISS  
- **Database / Cache**: PostgreSQL, Redis  
- **Deployment**: Docker + Docker Compose  
- **Auth**: Spotify OAuth2

---

## 📦 Getting Started

### 1. Clone the repo
```Bash
git clone https://github.com/THuynh-91/music-recommendation-spotify.git
cd music-recommendation-spotify
```

### 2. Set up environment variables
Create a file called **.env** in the project root and add your Spotify credentials:

```Bash
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

> Tip: You can also copy .env.example into .env and fill in your values.

### 3. Run with Docker
Build and start the containers with:

```Bash
docker compose up --build
```

The app will be available at [http://localhost:3000](http://localhost:3000).

---

## 📚 Roadmap
- [ ] User accounts + persistent preferences  
- [ ] More advanced recommendation strategies (clustering, embeddings)  
- [ ] Deploy to cloud (Render / Vercel / AWS)  

---

## 🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to improve.

---

## 📜 License
MIT License © 2025 [Tri Huynh](https://github.com/THuynh-91)
