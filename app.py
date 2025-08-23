import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time

# Title

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2021/03/chaitali-majumder/house-price-497112-KhCJQICS.jpg')



st.header('Model of housing prices to predict median house values in California ',divider=True)

#st.subheader('''User Must Enter Given values to predict Price:
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')


st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20230804/pngtree-an-upside-graph-showing-prices-and-houses-in-the-market-image_13000262.jpg')



temp_df = pd.read_csv('california.csv')

random.seed(30)

all_values=[]

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value=ss.transform([all_values])  




with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt=pickle.load(f)

price=chatgpt.predict(final_value)[0]


value=0

st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))
progress_bar=st.progress(value)
placeholder=st.empty()
placeholder.subheader('Predicting Price')
place=st.empty()
place.image('https://cdn.dribbble.com/userupload/23747162/file/original-77e7f1d34d9dfe2f8d372b0638306717.gif',width=80)





if price>0:
    
    for i in range (100):
        time.sleep(0.05)
        progress_bar.progress(i+1)

        
    body= f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    # st.subheader(body)

    st.success(body)

else:
    body='Invalid House Feaature Values'
    st.warning(body)

st.markdown('Designed by:**Fazal Rizvi**')
