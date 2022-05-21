import requests as rq
from bs4 import BeautifulSoup
import pandas as pd

def Scraper(id):
    url = f"https://www.imdb.com/title/{id}/reviews?ref_=tt_urv"
    request = rq.get(url)
    review_page = BeautifulSoup(request.content, "html.parser")
    reviews = review_page.findAll('div', {'class':'text show-more__control'})
    re = []
    for i in reviews:
        re.append(str(i))
    df_review = pd.DataFrame(re, columns = ['review'])
    return df_review, df_review