
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64

import defs_pred


st.set_page_config(
    page_title = 'Article category prediction app',
    page_icon = '💚',
    layout = 'wide'
)

# Logo 
st.image('cartoon_logo.jpg', use_column_width=True)


# Title 
st.write("""

#### This app predicts category for National Geoghraphic articles.


""")


# Sidebar
st.sidebar.write("## Get prediction for single article")
st.sidebar.text_input("", key="article", help="Enter single article", max_chars=10000)
col1, col2, col3 = st.sidebar.columns(3)
with col2:
    button1 = st.button('Predict')
st.sidebar.write("## Get predictions for more articles from CSV file")
st.sidebar.file_uploader("", key="file", help="Upload CSV file with 'Content' column")
col4, col5, col6 = st.sidebar.columns([1,2,1])
with col5:
    button2 = st.button('Predict more')

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download</a>'
    return href


if button1:
	article = st.session_state.article
	if article:
		with st.spinner("Calculating prediction..."):	
			#result from neural network
			prediction = defs_pred.predict(article)
			result = defs_pred.get_category(prediction)

		st.header('Input article')
		st.markdown(f"<h1 style='text-align: justify; font-size : 15px; color: gray'>{article}</h1>", unsafe_allow_html=True)
		st.header('Predicted category')
		for r in result:
			st.markdown(f"<h1 style='text-align: center; font-size : 30px; color: green;'>{r}</h1>", unsafe_allow_html=True)
	else:
		st.error('Invalid Article!')

elif button2:
	articles_file = st.session_state.file
	if articles_file:
		df_articles = pd.read_csv('test_data.csv')

		with st.spinner("Calculating predictions..."):		
			predictions = defs_pred.predict_more(df_articles)
			result = defs_pred.get_category_more(predictions)
		print(result)

		predictions = pd.DataFrame(pd.Series(result), columns=['category'])
		st.header('Input')
		st.write(df_articles)
		st.header('Predicted values')
		st.write(predictions)
		st.markdown(file_download(predictions), unsafe_allow_html=True)
	else:
		st.error('Invalid file!')
  
else:
	st.info('Upload input data in the sidebar to start!')




