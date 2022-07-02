#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shap
import pickle
import kmodes
from catboost import *
from PIL import Image
import streamlit as st

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[ ]:


loaded_model = pickle.load(open('assets/catboost_N.pickle', 'rb'))
loaded_model_P = pickle.load(open('assets/catboost_P.pickle', 'rb'))
loaded_model_K = pickle.load(open('assets/catboost_K.pickle', 'rb'))


# In[ ]:


#N
with open('assets/explainer_N.pkl', 'rb') as f:
     explainer = pickle.load(f)
with open('assets/shap_explainer_N.pkl', 'rb') as f:
     shap_explainer = pickle.load(f)
        
#P
with open('assets/explainer_P.pkl', 'rb') as f:
     explainer_P = pickle.load(f)
with open('assets/shap_explainer_P.pkl', 'rb') as f:
     shap_explainer_P = pickle.load(f)
        
#K
with open('assets/explainer_K.pkl', 'rb') as f:
     explainer_K = pickle.load(f)
with open('assets/shap_explainer_K.pkl', 'rb') as f:
     shap_explainer_K = pickle.load(f)


# In[ ]:


image = Image.open('assets/irri.jpg')

col1, col2 = st.columns([2,10])
with col1:
    st.image(image, width=100)
with col2:
    st.title('RCM Fertilizer Recommendation')
    
st.header('Enter information on farmer practices')


# In[ ]:


# carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)

cluster = st.selectbox('Cluster membership', [0,1,2,3,4,5,6,7])

season = st.radio('When is rice sown in the seedbed?',
                                 ['Wet season', 'Dry season'])
    
st.markdown("""
<style>
.small-font {
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="small-font">How many sacks of fresh, unmilled rice do you'
                       ' normally harvest from your farm lot during'
                       ' harvest in (date) using (type) variety?'
                       'Give your total harvest.</p>', unsafe_allow_html=True)

yield_1 = st.number_input('Number of sacks', value=0)

yield_2 = st.number_input('Weight of sack (kg)', value=0)

method = st.selectbox('How was rice harvested in the immediate previous season?',
                                 ['Combined', 'Manual', 'Reaper'])


# In[ ]:


if st.button('Show Fertilizer ranges'):
    test = [yield_1*yield_2, season, method, cluster]
    N_range = 4.50
    P_range = 1.37
    K_range = 2.58
    
    p_N = loaded_model.predict(test)
    p_P = loaded_model_P.predict(test)
    p_K = loaded_model_K.predict(test)

    st.success(f'Nitrogen (N): {p_N-N_range:.2f} - {p_N+N_range:.2f} kg/ha')
    st.success(f'Phosphorus (P): {p_P-P_range:.2f} - {p_P+P_range:.2f} kg/ha')
    st.success(f'Potassium (K): {p_K-K_range:.2f} - {p_K+K_range:.2f} kg/ha')
    
if st.button('Show SHAP plots'):
    test = [yield_1*yield_2, season, method, cluster]
    X_ho = pd.DataFrame(test).T
    col_names = ['RCM Normal Yield t/ha in 14% MC',
             'Growing season', 
             'RCM Harvesting Method',
             'cluster']
    X_ho.columns = col_names
    figN = shap.force_plot(base_value=shap_explainer.base_values[1],
                               shap_values= explainer.shap_values(X_ho),
                               feature_names=col_names,
                               out_names='Nitrogen',
                               matplotlib=True,
                               figsize=(10, 3),
                               show=False
                           )
    figP = shap.force_plot(base_value=shap_explainer_P.base_values[1],
                               shap_values= explainer_P.shap_values(X_ho),
                               feature_names=col_names,
                               out_names='Phosphorus',
                               matplotlib=True,
                               figsize=(10, 3),
                               show=False
                           )
    figK = shap.force_plot(base_value=shap_explainer_K.base_values[1],
                               shap_values= explainer_K.shap_values(X_ho),
                               feature_names=col_names,
                               out_names='Potassium',
                               matplotlib=True,
                               figsize=(10, 3),
                               show=False
                           )
    st.success('SHAP plots show the contribution of each farm characteristic to '
            "the prediction. Red bars indicate an increase from the model's average value, "
             'while blue bars indicate a decline. '
             'Actual prediction is in bold.')
    st.pyplot(figN, use_container_width=True)
    st.pyplot(figP, use_container_width=True)
    st.pyplot(figK, use_container_width=True)

