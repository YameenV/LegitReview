from preproces import Preprocessing
import pandas as pd
import streamlit as st
import pickle
from scraper import Scraper


st.title("Hello")
movieId = st.text_input(label="IMDB ID")
subBut = st.button("Submit")

if subBut:
    
    df, df_org = Scraper(movieId)

    try:
        data = Preprocessing(df)
    except:
        print("Could Not PreProcess")


    with open("logisticModel500","rb")as file:
        reviewModel = pickle.load(file)

    prediction = reviewModel.predict(data[0:100])
    df_pred = pd.DataFrame(prediction, columns =["prediction"])
    df_final = pd.concat([df, df_pred],axis = 1)

    positive = 0
    negative = 0
    for i, y in enumerate(df_org["review"]):
        if df_final["prediction"][i] == 1:
            st.text("Positive "+":- "+f'{y}')
            positive += 1
        else:
            st.text("Negative "+":- "+f'{y}')
            negative += 1

    st.header(f'Total Postitve score {positive}')
    st.header(f'Total Negative score {negative}')

   

    


