from fastapi import FastAPI
from pydantic import BaseModel
from utils import get_specific_phrase, preprocess_text, check_owner_by_naive_method, split_count
from pyvi import ViTokenizer, ViPosTagger
from constant import Threshold

import pandas as pd

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
based_tags = list(set(tag_df['phrase'].tolist()))
specific_phrases = get_specific_phrase()

@app.post("//house-owner-checking")
def check_house_owner(body: HouseOwnerCheckModel):

    data = dict(body)
    text = preprocess_text(data['description'])


    word_count = len(text.split(" "))
    if word_count < Threshold.WORD_COUNT_THRESHOLD:
        return {
            "is_owner": 1
        }


    if check_owner_by_naive_method(text) == 0:
        return {
            "is_owner": 0
        }


    emoji_count, word_count = split_count(text)
    print("emoji_count:", emoji_count)

    if emoji_count >= Threshold.EMOJI_COUNT_THRESHOLD:
        return {
            "is_owner": 0
        }


    fine = ''
    for c in list(text):
        if c.isalpha() or c == ' ':
            fine += c

    a = ViTokenizer.tokenize(fine)
    tag = ViPosTagger.postagging(a)
    tag_df = pd.DataFrame(columns = ['phrase', 'tag'])
    tag_df['phrase'] = tag[0]
    tag_df['tag'] = tag[1]
    tag_df = tag_df[tag_df['phrase'].str.contains('_')]


    selected_tags = list(set(tag_df['phrase'].tolist()))

    selected_words = [item.replace("_", " ").strip() for item in selected_tags]

    sum = 0
    for item in selected_words:
        for phrase in specific_phrases:
            if item in phrase:
                sum += 1
                print(item)
                break
    if sum >= Threshold.SPECIFIC_PHRASE_COUNT_THRESHOLD:
        return {
            "is_owner": 0
        }

    sum = 0
    for item in selected_tags:
        for based_tag in based_tags:
            if item in based_tag:
                sum += 1
                print(item)
                break

    print(sum)
    if sum > Threshold.VIVID_TAG_COUNT_THRESHOLD:
        return {
            "is_owner": 0
        }

    return {
            "is_owner": 1
        }