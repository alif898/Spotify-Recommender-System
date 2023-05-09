"""
Module containg functions to interact with Spotify API to generate recommendations.

Functions prefixed with '_' are meant to be private.

To build a model, get recommendations and add them to playlist, the following flow is used:
retrain_model -> generate_recommendations -> add_to_playlist
"""

# Importing required packages
from __future__ import annotations

import polars as pl
import random
import spotipy

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Iterable, List, Tuple


# Non-features to exclude when reading audio features JSON
exclusions = {'type', 'id', 'uri', 'track_href', 'analysis_url'}

# Track features, except artist dummy variables
track_features = {
    'track_popularity'
    , 'danceability'
    , 'energy'
    , 'key'
    , 'loudness'
    , 'mode'
    , 'speechiness'
    , 'acousticness'
    , 'instrumentalness'
    , 'liveness'
    , 'valence'
    , 'tempo'
    , 'duration_ms'
    , 'time_signature'
}

# Features that require scaling, as determined during EDA
scale_features = [
    'track_popularity'
    , 'key'
    , 'loudness'
    , 'tempo'
    , 'duration_ms'
    , 'time_signature'
]


# Note that private functions are prefixed with single underscore, _


def link_to_uri(link: str) -> str:
    """
    Function that fetches the URI from a playlist link.

    :param link: Playlist link
    :return: URI of input playlist
    """
    return link.split('/')[-1].split('?')[0]


def redirect_uri_to_port(link: str) -> int:
    """
    Function that gets port number from the redirect link.

    :param link: Format should be 'http://127.0.0.1:XXXX/'
    :return: XXXX as integer
    """
    if link[-1] == '/':
        link = link[:-1]
    return int(link.split(':')[-1])


def _process_track(track: dict, sp: spotipy.client.Spotify, schema: List[str] | None = None) -> dict | None:
    """
    Function to extract features from a track JSON, including basic details, audio features and artists.

    :param track: Dictionary representing track JSON, structure as extracted via Spotify API
    :param sp: Spotipy client, with authentication manager included
    :param schema: Fixed schema is used at prediction time, to ensure features are same as training data
    :return: Dictionary with feature name, value as key, value pair for that track
    """
    result = {}
    try:
        # Fetching track URI
        track_uri = track['uri']
        if 'local' in track_uri or track_uri is None:
            # 'Local' tracks do not have audio features, we will ignore them
            return
        result['track_uri'] = track_uri

        # Fetching track name and popularity
        track_name = track['name']
        result['track_name'] = track_name
        result['track_popularity'] = track['popularity']

        # Fetching audio features requires a separate API call based on track URI
        next_audio_feature = sp.audio_features(tracks=track_uri)[0]
        for feature, value in next_audio_feature.items():
            if feature not in exclusions:
                result[feature] = value

        # Each track may have multiple artist, we will one-hot encode artist names as additional features
        artists = list(map(lambda t: t['name'], track['artists']))
        # If schema is provided, check if artists in schema are present in current track
        if schema:
            for feature in schema:
                # Checking if feature is not a non-artist feature
                if feature not in track_features:
                    if feature in artists:
                        result[feature] = 1
                    else:
                        result[feature] = 0
        # If not, just add artists of current track
        else:
            for artist in artists:
                result[artist] = 1

    except (TypeError, ValueError) as e:
        # Ignore tracks that can't be read and print the corresponding error to aid in debugging
        print(f"Error in processing: {track}: {e}")

    return result if result else None


def _track_collection_to_df(
    track_collection: Iterable[dict]
    , sp: spotipy.client.Spotify
    , schema: List[str] | None = None
) -> pl.DataFrame:
    """
    Function that iterates through tracks in a playlist, extracts relevant features and returns a DataFrame.
    Each row represents one track.

    :param track_collection: Iterable (Playlists, user top tracks) containing dictionaries each representing a track
    :param sp: Spotipy client, with authentication manager included
    :param schema: Fixed schema is used at prediction time, to ensure features are same as training data
    :return: Polars DataFrame
    """
    # Initializing list of dictionaries to store track data
    values = []

    # Iterate through each track in the collection
    for track_json in track_collection:
        # For some reason, empty track_json may be collected
        if track_json:
            next_track = _process_track(track=track_json, sp=sp, schema=schema)
            if next_track:
                values.append(next_track)

    df = pl.DataFrame(values)
    # Fill empty feature values with 0, meant for artist dummy variables
    df = df.fill_null(strategy='zero')

    return df


def retrain_model(
    sp: spotipy.client.Spotify
    , n_playlists: int = 20
    , randomize: bool = True
    , seed: int = 1
) -> Tuple[ColumnTransformer, NearestNeighbors, pl.DataFrame, List[str]]:
    """
    Function that trains NearestNeighbors model from scratch.
    Pool of tracks to draw recommendations from will be built from playlists made by Spotify.

    :param sp: Spotipy client, with authentication manager included
    :param n_playlists: Number of playlists to use, each generally contains 50 tracks, default is 20 playlists, max is 50.
    :param randomize: Whether to randomize which playlist to fetch from Spotify, default is true
    :param seed: Random seed, default is 1
    :return: Fitted ColumnTransformer that handles standardization, fitted NearestNeighbors, DataFrame containing tracks used in training, schema of DataFrame used in training
    """

    # Extract curated playlists created by Spotify itself
    spotify_playlists = sp.user_playlists('spotify')
    playlist_ids = spotify_playlists['items']

    if randomize:
        # Set random seed and shuffle
        random.seed(seed)
        random.shuffle(playlist_ids)

    df = pl.DataFrame()
    # Process each playlist individually and collect tracks
    for i in range(n_playlists):
        next_id = playlist_ids[i]['id']
        # Extracting iterable of track JSON from playlist
        playlist_iterable = map(lambda t: t['track'], sp.playlist_items(next_id)['items'])
        # Convert to DataFrame
        next_df = _track_collection_to_df(track_collection=playlist_iterable, sp=sp)
        # Concatenate results
        df = pl.concat([df, next_df], how='diagonal')

    # Check for duplicate tracks
    df = df.unique(subset=['track_uri'])
    # Fill empty feature values with 0, meant for artist dummy variables
    df = df.fill_null(strategy='zero')

    # When training the model, we exclude the track URI and name
    # We recover this during prediction time to make recommendations
    train_df = df.select(pl.exclude('track_uri', 'track_name'))

    # Unfortunately, sci-kit learn works with Pandas DataFrames better
    train_df_pd = train_df.to_pandas()

    # Here we apply standardization to selected continuous features
    ct = ColumnTransformer([('scaler', StandardScaler(), scale_features)], remainder='passthrough')
    ct.fit(train_df_pd)
    train_df_pd_scaled = ct.transform(train_df_pd)

    # Here we fit the NearestNeighbors algorithm with our training data
    nn = NearestNeighbors()
    nn.fit(train_df_pd_scaled)

    # Schema is extracted to use at prediction time
    df_schema = list(train_df.schema.keys())

    # Add row number to original DataFrame to retrieve recommendations later
    df = df.with_row_count(name='row_number')

    return ct, nn, df, df_schema


def generate_recommendations(
    ct: ColumnTransformer
    , nn: NearestNeighbors
    , df: pl.DataFrame
    , schema: List[str]
    , sp: spotipy.client.Spotify
    , n_recommendations: int = 50
) -> pl.DataFrame:
    """
    Function to generate recommendations for current user.
    User top tracks will be fetched and recommendations will be drawn from fitted NearestNeighbors model.

    :param ct: Fitted ColumnTransformer that handles standardization of selected features
    :param nn: Fitted NearestNeighbors model
    :param df: DataFrame containing tracks used in training
    :param schema: Schema of DataFrame used in training, to ensure features match between training data and new data
    :param sp: Spotipy client, with authentication manager included
    :param n_recommendations: Number of track recommendations to generate, default is 50
    :return: Polars DataFrame containing recommended tracks as provided by NearestNeighbors model
    """
    # Fetching the top n tracks of user
    top_tracks = sp.current_user_top_tracks(limit=n_recommendations)['items']
    # Here we reuse the schema used in the training set to ensure features are the same
    top_tracks_df = _track_collection_to_df(track_collection=top_tracks, sp=sp, schema=schema)

    # Exclude non-feature columns
    top_tracks_df = top_tracks_df.select(pl.exclude('track_uri', 'track_name'))
    # Unfortunately, sci-kit learn works with Pandas DataFrames better
    top_tracks_df_pd = top_tracks_df.to_pandas()

    # Using ColumnTransformer to standardize data
    top_tracks_df_pd_scaled = ct.transform(top_tracks_df_pd)
    # Using NearestNeighbors to fetch recommendations, we will get a list of indexes
    recs_idx_nested = nn.kneighbors(top_tracks_df_pd_scaled, n_neighbors=1, return_distance=False)
    # Result is list of lists, need to flatten first
    recs_idx = [idx[0] for idx in recs_idx_nested]

    # Check for duplicate recommendations
    recs_idx = list(set(recs_idx))
    # Filter based on indexes of closest songs
    recommendations = df.filter(pl.col('row_number').is_in(recs_idx))

    return recommendations


def add_to_playlist(
    df: pl.DataFrame
    , sp: spotipy.client.Spotify
    , name: str = 'Recommendations by Alif'
    , desc: str = 'Thank you for trying my recommender system. Visit me at https://github.com/alif898!'
) -> str:
    """
    Function to add recommended tracks to user account as new playlist.

    :param df: DataFrame containing songs to add
    :param sp: Spotipy client, with authentication manager included
    :param name: Name of new playlist to create
    :param desc: Description for new playlist
    :return: Link to the playlist created
    """
    # Get the track URI
    tracks = df['track_uri'].to_list()
    # Initialize empty playlist
    user = sp.current_user()['uri'].removeprefix('spotify:user:')
    playlist = sp.user_playlist_create(user=user, name=name, description=desc)
    playlist_uri = playlist['id']

    # Spotify API only allows addition of 100 tracks per request
    for i in range(0, len(tracks), 100):
        next_batch = tracks[i:i + 100]
        sp.playlist_add_items(playlist_id=playlist_uri, items=next_batch)

    # Returning link to the playlist that was just created
    return sp.playlist(playlist_id=playlist_uri)['external_urls']['spotify']


if __name__ == "__main__":
    pass
