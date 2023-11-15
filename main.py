from fastapi import FastAPI, Depends
from typing import Union
from pydantic import BaseModel
from utils import preprocess_text, check_owner_by_naive_method, split_count
from pyvi import ViTokenizer, ViPosTagger
import pandas as pd
import numpy as np

app = FastAPI(
    title='FastAPI House Owner Checking', docs_url='/docs',
    dependencies=[
        # Depends(get_query_token),
        # Depends(reusable_oauth2)
    ]
)

@app.get("//healthcheck")
def healthcheck():
    """Function checking health"""
    return {
        "data": "200"
    }

class HouseOwnerCheckModel(BaseModel):
    description: str
    phone: str


tag_df = pd.read_csv('./data/selected_neg_tag.csv')
based_tags = tag_df['phrase'].tolist()

@app.post("//house-owner-checking")
def check_house_owner(body: HouseOwnerCheckModel):

    data = dict(body)

    text = preprocess_text(data['description'])
    if check_owner_by_naive_method(text) == 0:
        return 0


    emoji_count, word_count = split_count(text)
    print("emoji_count:", emoji_count)

    if emoji_count >= 3:
        return 0

    fine = ''
    for c in list(text):
        if c.isalpha() or c == ' ':
            fine += c

    a = ViTokenizer.tokenize(fine)
    tag = ViPosTagger.postagging(a)
    tag_df = pd.DataFrame(columns = ['phrase', 'tag'])
    tag_df['phrase'] = tag[0]
    tag_df['tag'] = tag[1]
    tag_df = tag_df[tag_df['tag'] == 'A']

    selected_tags = tag_df['phrase'].tolist()
    cnt_arr = np.array([tag in based_tags for tag in selected_tags])
    sum = np.sum(cnt_arr)

    if sum > 10:
        return 0
    print(sum)

    return 1