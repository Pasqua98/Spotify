import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import warnings
import seaborn as sns
import kagglehub
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import lyricsgenius
from transformers import pipeline
import numpy as np
import csv
import random

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


def clean_lyrics(text):
    text = re.sub(r"\[.*?\]", "", text)  # remove [Chorus], etc.
    text = text.lower()
    return text.strip()

def extract_axes(lyrics):
    lyrics = clean_lyrics(lyrics)
    vader = SentimentIntensityAnalyzer()

    emotion_classifier = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True
    )

    # Emotion scores
    emotions = emotion_classifier(lyrics[:512])[0]
    emo = {e["label"]: e["score"] for e in emotions}

    # Sentiment
    sent = vader.polarity_scores(lyrics)

    # --- Exuberance ---
    exuberance = (
        emo.get("excitement", 0) +
        emo.get("joy", 0) +
        emo.get("optimism", 0) +
        max(sent["compound"], 0)
    )

    # --- Containment (tension) ---
    containment = (
        emo.get("anger", 0) +
        emo.get("fear", 0) +
        emo.get("sadness", 0) +
        abs(min(sent["compound"], 0))
    )

    # Normalize to [-1, +1]
    exuberance = np.tanh(exuberance)
    containment = np.tanh(containment)

    return exuberance, containment

def map_playlist(row, centers):
    c = centers.loc[row.cluster]

    if c.exuberance > 0.4 and c.containment > 0.4:
        return "Gym"
    if c.exuberance > 0.4 and c.containment < 0:
        return "Morning Energy"
    if c.exuberance < 0 and c.containment > 0.3:
        return "Night Chill"
    if c.exuberance < 0 and c.containment < 0:
        return "Study Focus"
    return "Driving"

def process_songs(rows,track,genius):


        song = genius.search_song(track['name'], track['artist'])
        lyrics = song.lyrics if song else ""
        # genres = track.get("genres", [])

        if lyrics:
            exuberance, containment = extract_axes(lyrics)
        else:
            # fallback to genre priors (optional)
            exuberance = 0.0
            containment = 0.0

        rows.append({
            "track_uri": track["uri"],
            "track_name": track["name"],
            "artist": track["artist"],
            "exuberance": round(exuberance, 4),
            "containment": round(containment, 4),
            "lyrics_available": int(lyrics is not None)
        })




if __name__ == "__main__":

    if not os.path.exists("song_mood_axes.csv"):
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


        all_tracks = random.sample(all_tracks, 100)

        genius = lyricsgenius.Genius(
            "SkclVT1F38CfxJiIUkwt_kc06NQfHXrAvhEia1oqttCBX5qa3P9_2lbfZU4q0inr",
            timeout=20,  # ← increase timeout
            retries=3,  # ← retry on failure
            sleep_time=1,  # ← pause between retries
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)"],
            remove_section_headers=True
        )

        rows = []

        for i,track in enumerate(all_tracks):
            print("%s\n",i)
            process_songs(rows,track,genius)
            time.sleep(2)

        df = pd.DataFrame(rows)

        df.to_csv(
            "song_mood_axes.csv",
            index=False,
            encoding="utf-8"
        )
        with open("song_mood_axes.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    df = pd.read_csv("song_mood_axes.csv")

    import matplotlib.pyplot as plt

    plt.scatter(df["exuberance"], df["containment"], alpha=0.6)
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel("Exuberance")
    plt.ylabel("Containment")
    plt.title("Song Mood Space (Thayer)")
    plt.show()

    import numpy as np
    from sklearn.mixture import GaussianMixture

    X = df[["exuberance", "containment"]].values

    bic_scores = {}
    for k in range(2, 9):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42
        )
        gmm.fit(X)
        bic_scores[k] = gmm.bic(X)

    best_k = min(bic_scores, key=bic_scores.get)
    print("Best k by BIC:", best_k)

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=42,
        n_init=10
    )

    df["cluster"] = gmm.fit_predict(X)

    probs = gmm.predict_proba(X)

    df["confidence"] = probs.max(axis=1)

    centers = pd.DataFrame(
        gmm.means_,
        columns=["exuberance", "containment"]
    )

    print(centers)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        df.exuberance,
        df.containment,
        c=df.cluster,
        cmap="tab10",
        alpha=0.7
    )

    plt.scatter(
        centers.exuberance,
        centers.containment,
        c="black",
        s=150,
        marker="X"
    )

    plt.axhline(0, color="gray")
    plt.axvline(0, color="gray")

    plt.xlabel("Exuberance")
    plt.ylabel("Containment")
    plt.title("GMM Mood Clusters")
    plt.show()





