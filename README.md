# Spotify-Recommender-System

Get your recommendations [here](https://spotify-recommender.herokuapp.com/)!

Update: I have not been given permission from Spotify to progress the app further from development mode.
To try the recommender system, you may contact me directly and I can register you manually.

App will be kept running until my Heroku trial credits run out :(

## Introduction

Previously, I tried to build a POC recommender system for Spotify [here](https://github.com/alif898/Spotify-Classification-Recommendation-System), however that project had a few issues.
Many of the limitations arose from the use of Databricks Community Edition, most importantly, having no easy way to deploy the trained model.
I decided to take up the idea again, but this time with a new methodology and a focus on ensuring that the system is deployed, so that others are able to actually try my recommendations.

## Methodology

The method used to generate recommendations is [content-based filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering). This approach makes recommendations based on the attributes or features of items and the user's preferences. 
The idea is that if a user has shown a preference for certain features in the past, they are likely to prefer items that have similar features in the future.

Here we are able to fetch a user's past preferences with the Spotify API, which will allow us to get a user's top tracks.

Next, we want to look for a pool of tracks to source recommendations from.
We will use the curated playlists created by the [Spotify](https://open.spotify.com/user/spotify?si=67a52dad1d244a8a) to achieve this.
We will fetch 50 playlists, each with 50 tracks, giving us a total of about 2500 tracks.


From the Spotify API, we are able to fetch a wide variety of information about a track. 
Besides information such as artist name, track popularity, there is a particular interesting category of variables known as audio features. More details [here](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features).

Essentially, these audio features capture characteristics of tracks that might seemingly be difficult to quantify. 
For example, the 'energy' variable, which goes from 0-1, represents the intensity and energy that a track has. 
According to Spotify, energetic tracks feel fast, loud and noisy. 
Unfortunately, technical details of how such variables are actually calculated are not revealed (it is unlikely that Spotify measures this manually). 
Nevertheless, by being able to quantify seemingly abstract characteristics of tracks, we can take advantage of these features to build our recommender system.
We will also use the artists as features, by using a dummy variable representation, it is likely that people will like a track by an artist if they already like other tracks from the same artist.
Note that we are only able to use the artists present in the training data.
It is possible that a user's top tracks contain artists that were not present in the training data.

There are many ways to look for items with similar features in content-based filtering, but here we use K-Nearest Neighbors (KNN).
This is a simple distance-based approach and here we use k=1.
As KNN is not scale invariant, some of the features that are not already within 0-1 will be standardized.
As such the idea is to fetch a user's top tracks and use KNN to get similar tracks from our pool of potential recommendations.

By giving them similar tracks to those that they already like, we are likely able to deliver recommendations that they will enjoy.

You may ask? If we train the model once, won't the recommendations be stale, using the same old tracks?
In order to handle this, we will retrain the model occasionally.
In terms of implementation, we will keep track of the date the model was trained.
If it has been 7 days, the model will be retrained.
This also ensures that users can revisit my recommender system again to get new recommendations.

## Implementation

```spotify_utilities.py``` contains all the functions, such as data processing for the playlists to extract the features we need, along with the training of the KNN model and the generation of recommendations.
Note that we use [Polars](https://github.com/pola-rs/polars) here instead of pandas. 
Polars is an up-and-coming pandas alternative that is much faster.
It uses Apache Arrow, it has lazy evaluation and is able to handle larger than RAM data.
Although the scale of our system is unlikely to require such performance, I wanted to take the chance to experiment and learn Polars.

```app.py``` contains the code for the Flask app itself, including the login flow. 
The user will be prompted to sign-in and give permissions.
Once done, we will then load our models and check if it needs to be retrained.
Then we deliver recommendations and add them directly to a new playlist.
The model, including the ```StandardScaler```, ```NearestNeighbors``` objects, are stored in a ```.pkl``` file.
We also store a ```datetime``` object to keep track of when the model was trained. 
The size is too large to upload to GitHub, so I have zipped it.

We deploy our Flask app as a Docker container on Heroku, as specified in ```Dockerfile```.
Note that the Spotify API login details are meant to be in a ```.env``` file that is not uploaded for security reasons.
A sample file named ```sample.env``` shows how to set it up.
Sample instructions can be found [here](https://medium.com/analytics-vidhya/dockerize-your-python-flask-application-and-deploy-it-onto-heroku-650b7a605cc9).

## Conclusion

I am pleased to be able to finally implement my idea of delivering a seamless recommender system on Spotify.
We successfully built on the previous project and this time, we were able to implement a more robust system that uses content-based filtering.
Furthermore, we were able to deploy the recommender system. With a single link and login prompt, we were able to deliver the recommendations conveniently for the user.

Nevertheless, there is still room for improvement. Firstly, there are many other ways to build a content-based filtering system, besides KNN.
It may be worthwhile to try other methods and see if they work better.
Secondly, the UI is still simple, just a simple 'Sign-in' prompt and then the playlists is just created.
Lastly, there are other approaches to building recommender systems, such as collaborative filtering.
This [video](https://www.youtube.com/watch?v=pGntmcy_HX8&pp=ygUnc3BvdGlmeSByZWNvbW1lbmRhdGlvbiBzeXN0ZW0gYmxvb21iZXJn) from WSJ suggests that the actual Spotify recommender system uses collaborative filtering.

Thank you for visiting! Happy listening :)