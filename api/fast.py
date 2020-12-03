from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np
import requests as rq
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# import os
# from tensorflow.keras import models
# import joblib
# from content_lie_detector.encoders import CustomTokenizer
# from content_lie_detector.data import get_data


class Item(BaseModel):
    article_url: str


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "https://fakenews-ktojww43aq-ew.a.run.app",
    "http://fakenews-ktojww43aq-ew.a.run.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    # res = rq.get("https://breaking-bad-quotes.herokuapp.com/v1/quotes")
    # quote = res.json()[0]["quote"]
    # author = res.json()[0]["author"]
    # return f"""Coming Soon ...<br><p style="font-size:20px">{quote}</p><p><b>{author}<b></p>"""
    return {"status": "OK"}


@app.post("/isfakenews/")
async def create_item(item: Item):
    # TO DO: SCRAP URL TITLE + CONTENT

    # path = os.path.dirname(os.path.dirname(__file__)) + "/"
    # model = models.load_model(path + "model")
    # preprocessor = joblib.load(path + "preprocessor.joblib")
    # sample = preprocessor.transform(content)
    # pred = model.predict(sample)
    # return {"content": content,
    #         "proba_fake": round(pred.item(0), 2)}
    return {"url": item.article_url, "proba_fake": round(np.random.random(), 2)}
