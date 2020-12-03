from fastapi import FastAPI
import numpy as np
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
    "*"
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
    return {"article_url": item.article_url, "proba": round(np.random.random(), 2)}
