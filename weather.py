import math
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import datetime,time

#SET PAGE WIDE
st.set_page_config(page_title='Weather predictor',layout="centered")

#Get the ML model 

filename='model1.keras'
model = load_model(filename) 

#Title of the page with CSS

st.markdown("<h1 style='text-align: center; color: white;'> Weather predictor at Jena </h1>", unsafe_allow_html=True)

selected_date = st.date_input(
    "Select a date",
    datetime.now(),  # Default date
    min_value=datetime(2020, 1, 1),  # Minimum selectable date
    max_value=datetime(2025, 12, 31)  # Maximum selectable date
)


selected_month = selected_date.month
selected_day = selected_date.day
selected_year=selected_date.year

times = [time(hour, 0) for hour in range(24)]

# Select box for time selection
selected_time = st.selectbox(
    "Select an hour",
    times,
    index=datetime.now().hour
)


selected_hour = selected_time.hour
df = pd.read_csv('temp.csv', parse_dates=['Date Time'], index_col='Date Time')



filtered_df = df[ ((df.index.day == selected_day) &
                 (df.index.month == selected_month) &
                 (df.index.hour == selected_hour-1)) |  ((df.index.day == selected_day) &
                 (df.index.month == selected_month) &
                 (df.index.hour == selected_hour-2))  |  ((df.index.day == selected_day) &
                 (df.index.month == selected_month) &
                 (df.index.hour == selected_hour-3))  |  ((df.index.day == selected_day) &
                 (df.index.month == selected_month) &
                 (df.index.hour == selected_hour-4))  |  ((df.index.day == selected_day) &
                 (df.index.month == selected_month) &
                 (df.index.hour == selected_hour-5)) ]





df_as_np=filtered_df.to_numpy()
X=[]

for i in range(0,len(df_as_np)-5,5):
    row=[a for a in df_as_np[i:i+5]]
    X.append(row)
   
X=np.array(X)
# st.write(X)


if st.button("Predict Temperature via LSTM"):
    result=model.predict(X).flatten()
    mean_result=result.mean()
    st.success(f"Temperature will be {mean_result:.2f} (degC) on {selected_date} at {selected_time}")





filename2='model2.pkl'
model_rf = pickle.load(open(filename2,'rb')) 

df=pd.DataFrame({'month':[selected_month],'day':[selected_day],'year':[selected_year],'hour':[selected_hour]},index=[0])





if st.button("Predict Temperature via Random Forest"):
    result1=model_rf.predict(df)
   
    st.success(f"Temperature will be {result1[0]:.2f} (degC) on {selected_date} at {selected_time}")

