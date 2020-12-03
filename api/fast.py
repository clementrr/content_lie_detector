from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np
import requests as rq
# import os
# from tensorflow.keras import models
# import joblib
# from content_lie_detector.encoders import CustomTokenizer
# from content_lie_detector.data import get_data

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    res = rq.get("https://breaking-bad-quotes.herokuapp.com/v1/quotes")
    quote = res.json()[0]["quote"]
    author = res.json()[0]["author"]
    return f"""Coming Soon ...<br><p style="font-size:20px">{quote}</p><p><b>{author}<b></p>"""


@app.get("/isfakenews/{content}")
def predict(content):
    # path = os.path.dirname(os.path.dirname(__file__)) + "/"
    # model = models.load_model(path + "model")
    # preprocessor = joblib.load(path + "preprocessor.joblib")
    # sample = preprocessor.transform(content)
    # pred = model.predict(sample)
    # return {"content": content,
    #         "proba_fake": round(pred.item(0), 2)}
    return {"proba_fake": round(np.random.random(), 2)}
