import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import warnings
import seaborn as sns
import kagglehub
import torch

warnings.filterwarnings("ignore")

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

def get_data():

    sns.set_style("white")
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    return path

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

    import lyricsgenius

    genius = lyricsgenius.Genius(
        "SkclVT1F38CfxJiIUkwt_kc06NQfHXrAvhEia1oqttCBX5qa3P9_2lbfZU4q0inr",
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        remove_section_headers=True
    )

    song = genius.search_song("Numb", "Linkin Park")

    from transformers import pipeline

    emotion = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )

    lyrics = song.lyrics

    scores = emotion(lyrics)[0]

    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    for s in scores[:5]:
        print(s["label"], s["score"])
#build dataset in csv with emotion scores per each song, then cluster with superposition