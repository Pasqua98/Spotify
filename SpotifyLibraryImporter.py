import spotipy
from spotipy.oauth2 import SpotifyOAuth

def get_all_liked_tracks(sp):
    tracks = []
    results = sp.current_user_saved_tracks(limit=50)

    while results:
        for item in results["items"]:
            track = item["track"]
            if track:
                tracks.append({
                    "uri": track["uri"],
                    "id": track["id"],
                    "name": track["name"],
                    "artist": ", ".join(a["name"] for a in track["artists"]),
                    "album": track["album"]["name"]
                })
        results = sp.next(results) if results["next"] else None

    return tracks

def get_all_playlist_tracks(sp):
    tracks = []

    playlists = sp.current_user_playlists(limit=50)
    while playlists:
        for pl in playlists["items"]:
            playlist_name = pl["name"]
            results = sp.playlist_items(pl["id"], limit=100)

            while results:
                for item in results["items"]:
                    track = item.get("track")
                    if track:
                        tracks.append({
                            "uri": track["uri"],
                            "id": track["id"],
                            "name": track["name"],
                            "artist": ", ".join(a["name"] for a in track["artists"]),
                            "album": track["album"]["name"],
                            "playlist": playlist_name
                        })
                results = sp.next(results) if results["next"] else None

        playlists = sp.next(playlists) if playlists["next"] else None

    return tracks

def dedupe_by_uri(tracks):
    unique = {}
    for t in tracks:
        unique[t["uri"]] = t
    return list(unique.values())

import pandas as pd

def chunks(lst, n=100):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

features = []

def get_features(sp, track_uris):

    for batch in chunks(track_uris, 100):
        af = sp.audio_features(batch)
        for f in af:
            if f is not None:
                features.append(f)

    df_features = pd.DataFrame(features)
    return df_features


if __name__ == "__main__":

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            scope="user-library-read playlist-read-private playlist-read-collaborative",
            redirect_uri="http://127.0.0.1:8888/callback",
            cache_path=".spotify_token_cache",
            open_browser=True
        )
    )

    liked_tracks = get_all_liked_tracks(sp)
    playlist_tracks = get_all_playlist_tracks(sp)
    all_tracks = dedupe_by_uri(liked_tracks + playlist_tracks)
    uris=[u['uri'] for u in all_tracks]
    #get kaggle dataset, then analyze data from there. Function like data_analysis are not included in APIs anymore
    print("Features:", af)