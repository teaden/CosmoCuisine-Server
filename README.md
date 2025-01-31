# CosmoCuisine - Server

This repository contains FastAPI server code for the [CosmoCuisine iOS application](https://github.com/teaden/CosmoCuisine-Client). The primary function of the code is to provide an API for utilizing a Time Delay Neural Network (ECAPA-TDNN) trained on the VoxLingua107 data set to identify spoken language. The CosmoCuisine iOS application uses this functionality to recognize the user's spoken language and, from a user specified query, search for matching food items in a local database based on parsed keywords.

## Usage

* fastapi run fastapi_server.py
