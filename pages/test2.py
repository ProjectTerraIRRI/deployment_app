#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import kmodes
from PIL import Image
import streamlit as st

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[ ]:


with open('assets/kproto.pkl', 'rb') as f:
    model = pickle.load(f)
with open('assets/mmscaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
coordinates = pd.read_excel('assets/lat_long_grid.xlsx')


# In[ ]:


# For getting coordinates
df_coords = coordinates.copy()
df_coords['REGION'] = df_coords['REGION'].apply(lambda x: str(x))
df_coords = df_coords.set_index(['REGION', 'PROVINCE', 'MUNICIPALITY'])

# For getting dropdown values
loc_dict = (coordinates[['REGION', 'PROVINCE', 'MUNICIPALITY']].astype(str)
            .groupby(['PROVINCE', 'REGION'])
            .agg(list).reset_index()
            .groupby('REGION')[['PROVINCE', 'MUNICIPALITY']]
            .apply(lambda x: x.set_index('PROVINCE').to_dict(orient='index'))
            .to_dict()
            )
regions = ['1', '2', '3', '4A', '4B', '5', '6', '7', '8', '9', '10', '11',
           '12', '13', 'ARMM', 'CAR', 'NCR']


# In[ ]:


image = Image.open('assets/irri.jpg')

col1, col2 = st.columns([2,10])
with col1:
    st.image(image, width=100)
with col2:
    st.title('Cluster membership')


# In[ ]:


crops_year = st.selectbox('Crops per year', [1,2,3])

water_regime = st.radio('Water Regime',
                                 ['Irrigated', 'Rainfed'])
    
water_supply = st.selectbox('Water Supply Status', ['Adequate', 'Short', 'Submergence'])
crops = st.selectbox('Crop establishment',
                     ['Manually transplanted', 'Wet Seeded', 'Dry seeded',
                 'Mechanically transplanted'])

region = st.selectbox('Region', regions)
if region:
    provinces = list(loc_dict[region].keys())
    province = st.selectbox('Province', provinces)
    if provinces:
        municipalities = loc_dict[region][province]['MUNICIPALITY']
        municipality = st.selectbox('Municipality', municipalities)


# In[ ]:


if st.button('Assess'):
    sample_data = {0: {'lat': 0, 'lon': 0, 'elev': 0,
                       'Crops per year': crops_year,
                       'RCM Water Regime': water_regime,
                       'RCM Crop establishment': crops,
                       'RCM water supply status': water_supply
                       }}
    sample_data = pd.DataFrame(sample_data).T[sample_data[0].keys()]
    sample_data.loc[0, ['lat', 'lon', 'elev']] = list(
        df_coords
        .loc[region]
        .loc[province]
        .loc[municipality])
    sample_data[['lat', 'lon', 'elev']] = scaler.transform(
        sample_data[['lat', 'lon', 'elev']])
    cluster = model.predict(sample_data, list(range(4,7)))[0]
    if (cluster==5) & (water_regime=='Rainfed'):
        cluster=7
    st.success(f'Cluster: {cluster}')

