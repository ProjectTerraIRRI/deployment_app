#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')




# In[ ]:
recom = pd.read_csv('assets/recommender_table_v3.csv')

# For getting coordinates
df_coords = recom.copy()
df_coords['Location_Region'] = df_coords['Location_Region'].apply(lambda x: str(x))
df_coords = df_coords.set_index(['Location_Region', 'cluster', 'Growing season',
                                 'RCM Harvesting Method', 'yield'])

# For getting dropdown values
loc_dict = (recom[['Location_Region', 'cluster', 'Growing season',
                                 'RCM Harvesting Method', 'yield']].astype(str)
            .groupby(['RCM Harvesting Method', 'Growing season','cluster', 'Location_Region'])
            .agg(list).reset_index()
            .groupby('Location_Region')[['cluster', 'Growing season',
                                 'RCM Harvesting Method', 'yield']]
            .apply(lambda x: x.set_index(['cluster','Growing season',
                                          'RCM Harvesting Method']).to_dict(orient='index'))
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
    st.title('Sub-clusters')


# In[ ]:


region = st.selectbox('Region', regions)
if region:
    clusters = list(loc_dict[region].keys())
    cluster = st.selectbox('cluster', clusters)
    if clusters:
        seasons = loc_dict[region][cluster]['Growing season']
        season = st.selectbox('Growing season', seasons)


