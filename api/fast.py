import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras import models
from api.helpers import get_article_from_url,\
                        preprocess_and_predict,\
                        make_heatmap_html


# Init item for POST
class Item(BaseModel):
    article_url: str


# Create app
app = FastAPI()

# Set authorizations to accept all origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get preprocessor and model
path = os.path.dirname(os.path.dirname(__file__)) + "/"
model = models.load_model(path + "model")
preprocessor = joblib.load(path + "preprocessor.joblib")


# Endpoint: Health
@app.get("/")
def index():
    return {"status": "OK"}


# Endpoint Get Prediction
@app.get("/isfake/")
async def read_item(article_url: str):

    # Get text of Article
    txt = get_article_from_url(article_url)

    # Preprocess text and predict
    pred = preprocess_and_predict(txt, preprocessor, model)

    # Generate txt with heatmap as html
    heatmap_html = make_heatmap_html(txt, preprocessor, model)

    return {"proba": round(pred.item(0), 2),
            "heatmap_html": heatmap_html}


# Endpoint Post Url
@app.post("/isfakenews/")
async def create_item(item: Item):

    # Get text of Article
    txt = get_article_from_url(item.article_url)

    # Preprocess text and predict
    pred = preprocess_and_predict(txt, preprocessor, model)

    # Generate txt with heatmap as html
    heatmap_html = make_heatmap_html(txt, preprocessor, model)

    return {"proba": round(pred.item(0), 2),
            "heatmap_html": heatmap_html}
