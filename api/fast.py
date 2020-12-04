from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import trafilatura as tf
import os
from tensorflow.keras import models
import joblib
# import numpy as np
# from content_lie_detector.encoders import CustomTokenizer
# from content_lie_detector.data import get_data


# Init item for POST
class Item(BaseModel):
    article_url: str


# Create app
app = FastAPI()

# Set authorizations to accept all origins
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint: test
@app.get("/")
def index():
    return {"status": "OK"}


# Endpoint: prediction
@app.post("/isfakenews/")
async def create_item(item: Item):
    # Get article url
    article_url = item.article_url
    # Get Article, as json, from url
    r = tf.fetch_url(article_url)
    j = tf.extract(r, json_output=True)
    # Clean and Get Title + Content
    j = json.loads(j)
    txt = j.get("title", "") + " " + j["text"]
    txt = txt.replace("\n", "")

    # Get preprocessor and model
    path = os.path.dirname(os.path.dirname(__file__)) + "/"
    model = models.load_model(path + "model")
    preprocessor = joblib.load(path + "preprocessor.joblib")
    # Preprocess text and predict
    txt = preprocessor.transform([txt])
    pred = model.predict(txt)
    return {"proba": round(pred.item(0), 2)}
    # return {"article_url": item.article_url,
    #         "proba": round(np.random.random(), 2)}
