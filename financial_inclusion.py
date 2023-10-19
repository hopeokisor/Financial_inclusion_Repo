import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import pickle

#----------------------LOAD MODEL--------------------
model = pickle.load(open('Financial_inclusion.pkl', "rb"))

st.markdown("<h1 style = 'text-align: centre; color: #F11A7B'>FINANCIAL INCLUSION PROJECT</h1> ", unsafe_allow_html = True)
st.markdown("<h3 style = ''text-align: centre; text-align: right; color: #8062D6'>Built By Hope In GoMyCode Sanaith Wizard</h3>", unsafe_allow_html= True)

st.image('pngwing.com (11).png', width=700)

st.markdown("<h1 style = 'top_margin: 0rem; text-align: centre; color: #557A46'>PROJECT SUMMARY</h1>", unsafe_allow_html= True)

st.markdown("<p style = 'top_margin: 0rem; text-align: justify; color: #00C4FF'>Financial inclusion refers to the effort to provide individuals and businesses with access to affordable and appropriate financial products and services. The goal of financial inclusion is to ensure that all people, regardless of their socioeconomic status, have access to basic financial tools and resources to manage their financial lives, save for the future, invest in opportunities, and protect themselves against financial shocks.Financial inclusion is seen as a key driver of economic development and poverty reduction. It can help individuals and communities build assets, access opportunities, and improve their overall well-being. Governments, financial institutions, and international organizations often work together to promote financial inclusion through policy initiatives, regulatory changes, and the development of financial products and services tailored to the needs of underserved populations</p>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html= True)

username = st.text_input('Enter your name')
if st.button('submit name'):
    st.success(f"Welcome {username}. Pls use according to usage guidelines")

data = pd.read_csv('Financial_inclusion_dataset.csv')
st.write(data.sample(10))

st.sidebar.image('pngwing.com (12).png', caption= f'Welcome {username}')
input_type = st.sidebar.selectbox('Select Your Prefered Input Type', ['Slider Input', 'Number Input'])

['education_level', 'job_type', 'cellphone_access', 'household_size', 'country', 'relationship_with_head', 'location_type']
if input_type == 'Slider Input':
    education = st.sidebar.select_slider('education_level', data['education_level'].unique())
    job_type = st.sidebar.select_slider('job_type', data['job_type'].unique())
    cellphone = st.sidebar.select_slider('cellphone_access', data['cellphone_access'].unique())
    household = st.sidebar.slider('household_size', data['household_size'].min(), data['household_size'].max())
    country = st.sidebar.select_slider('country', data['country'].unique())
    relationship = st.sidebar.select_slider('relationship_with_head', data['relationship_with_head'].unique())
    location = st.sidebar.select_slider('location_type', data['location_type'].unique())

else:
    education = st.sidebar.selectbox('education_level', data['education_level'].unique())
    job_type = st.sidebar.selectbox('job_type', data['job_type'].unique())
    cellphone = st.sidebar.selectbox('cellphone_access', data['cellphone_access'].unique())
    household = st.sidebar.number_input('household_size', data['household_size'].min(), data['household_size'].max())
    country = st.sidebar.selectbox('country', data['country'].unique())
    relationship = st.sidebar.selectbox('relationship_with_head', data['relationship_with_head'].unique())
    location = st.sidebar.selectbox('location_type', data['location_type'].unique())

    st.markdown("<br>", unsafe_allow_html= True)

    # Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'education_level' : education, 'job_type' : job_type, 'cellphone_access' : cellphone, 'household_size' : household, 'country' : country, 'relationship_with_head' : relationship, 'location_type' : location}])
st.write(input_variable)

from sklearn.preprocessing import LabelEncoder, StandardScaler
encoder = LabelEncoder()
scaler = StandardScaler()

for i in input_variable:
    if input_variable[i].dtypes != 'O':
        input_variable[i] = scaler.fit_transform(input_variable[[i]])
    else:
        input_variable[i] = encoder.fit_transform(input_variable[i]) 

# Create a tab for prediction and interpretation
pred_result, interpret = st.tabs(["Prediction Tab", "Interpretation Tab"])
prediction = None
with pred_result:
    if st.button('PREDICT'):
        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_variable)
        st.write("Bank Account Status :", prediction)
        st.toast('Input is Predicted')
    else:
        st.write('Pls press the predict button for prediction')
    

with interpret:
    st.subheader('Model Interpretation')
    if prediction == 0:
        st.write(['This customer is not likely to open a bank account'])
    elif prediction == 1:
        st.write(['This customer is very likely to open a bank account'])    

    # st.write(f"CHURN = {model.round(2)} + {model.coef_[0].round(2)} education_level + {model.coef_[1].round(2)} job_type + {model.coef_[2].round(2)} cellphone_access + {model.coef_[2].round(2)} household_size + {model.coef_[2].round(2)} country + {model.coef_[2].round(2)} relationship_with_head + {model.coef_[2].round(2)} location_type")

    