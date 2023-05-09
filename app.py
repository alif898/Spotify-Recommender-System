# Importing required packages
from __future__ import annotations

import datetime
import flask
import os
import pickle

from flask import Flask, session, request, redirect
from flask_session import Session
from spotipy.oauth2 import SpotifyOAuth


# These are pre-defined functions that will help train the model and deliver recommendations
from spotify_utilities import *
# Defining scope when connecting to Spotify API
scope = ['playlist-modify-public', 'user-top-read']


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
Session(app)


@app.route('/')
def deliver_recommendations() -> str | flask.Response:
    """
    Main function that will deliver recommendations to user.
    User will be prompted to log in.
    Model will be loaded from pkl file to recommend tracks.
    If model was trained more than 7 days ago, it will be retrained.

    :return: Link to the playlist created containing recommended tracks, or login prompt
    """

    # Initialize cache handler
    cache_handler = spotipy.cache_handler.FlaskSessionCacheHandler(session=session)
    # Initialize authentication manager, note that client_id, client_secret and redirect_uri are set as .env variables
    auth_manager = SpotifyOAuth(
        scope=scope
        , cache_handler=cache_handler
        , show_dialog=True
    )

    # 2. After login, access token is fetched here
    if request.args.get('code'):
        auth_manager.get_access_token(request.args.get('code'))
        return redirect('/')

    # 1. Initially, user will be prompted to log in
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        auth_url = auth_manager.get_authorize_url()
        return f'<h2><a href="{auth_url}">Sign in</a></h2>, ' \
               f'<h2><a href="https://github.com/alif898">Visit my GitHub!</a></h2>'

    # After login, we can then establish connection to Spotify API
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Set folder path for model
    pickle_path = os.path.join(app.root_path, 'model.pkl')
    # Load previously saved model
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    model_train_date, ct, nn, df, df_schema = model

    # Checking if model was trained more than 7 days ago
    now = datetime.datetime.now()
    time_diff = now - model_train_date
    # If yes, retrain model, this ensures recommendations are always fresh and relevant
    if time_diff.days >= 7:
        ct, nn, df, df_schema = retrain_model(sp, n_playlists=50)
        # Update model train date
        model_train_date = now

    # Generate recommendations and add them to playlist
    recs = generate_recommendations(ct, nn, df, df_schema, sp, n_recommendations=50)
    playlist_link = add_to_playlist(recs, sp)

    # Overwrite saved model, it may be a new model
    with open(pickle_path, 'wb') as f:
        model = model_train_date, ct, nn, df, df_schema
        pickle.dump(model, f)

    # Show user the link to the created playlist
    return f'<h2><a href="{playlist_link}">View the recommended songs here!</a></h2>, ' \
           f'<h2><a href="https://github.com/alif898/Spotify-Recommender">View the code here!</a></h2>'


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
