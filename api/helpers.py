import math
import json
import numpy as np
import trafilatura as tf
from tensorflow.keras import backend as K
from tensorflow import compat

compat.v1.disable_eager_execution()


def get_article_from_url(article_url):
    # Get Article, as json, from url
    r = tf.fetch_url(article_url)
    j = tf.extract(r, json_output=True)
    # Clean and Get Title + Content
    j = json.loads(j)
    txt = j.get("title", "")
    txt = txt.replace("\n", "")
    return txt


def preprocess_and_predict(txt, preproc, model):
    txt = preproc.transform([txt])
    pred = model.predict(txt)
    return pred


def make_heatmap_html(txt, preproc, model):

    pred = preprocess_and_predict(txt, preproc, model)

    pad = preproc.transform([txt])

    class_output = model.output[:]
    last_conv_layer = model.get_layer("conv1d_92")

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads)
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([pad])

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # find the ratio of the text vs the conv layer length
    norm_len = preproc.max_len/last_conv_layer.output_shape[1]
    html = ""

    if pred <= 0.4:
        word_span = "<span style='background-color:rgba(57,160,{},{})'>{} </span>"
    elif pred <= 0.7:
        word_span = "<span style='background-color:rgba({},104,83,{})'>{} </span>"
    else:
        word_span = "<span style='background-color:rgba({},0,1,{})'>{} </span>"

    for j, i in enumerate(txt.split()):
        html += word_span.format(heatmap[math.floor(j/norm_len)]*255,
                                 heatmap[math.floor(j/norm_len)]-0.3,
                                 i)

    return html
